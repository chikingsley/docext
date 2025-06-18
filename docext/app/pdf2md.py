from __future__ import annotations

import asyncio
import os # Added
import re
import time
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import gradio as gr
from loguru import logger # Added

from docext.core.client import MistralOCRClient # Added
from docext.core.pdf2md.pdf2md import convert_to_markdown_stream
from docext.core.utils import convert_files_to_images


def process_tags(content: str) -> str:
    content = content.replace("<img>", "&lt;img&gt;")
    content = content.replace("</img>", "&lt;/img&gt;")
    content = content.replace("<watermark>", "&lt;watermark&gt;")
    content = content.replace("</watermark>", "&lt;/watermark&gt;")
    content = content.replace("<page_number>", "&lt;page_number&gt;")
    content = content.replace("</page_number>", "&lt;/page_number&gt;")
    content = content.replace("<signature>", "&lt;signature&gt;")
    content = content.replace("</signature>", "&lt;/signature&gt;")

    return content


def pdf_to_markdown_ui(
    model_name: str, max_img_size: int, concurrency_limit: int, max_gen_tokens: int
):
    with gr.Row():
        with gr.Column():
            # Add status indicator for concurrent processing
            gr.Markdown(
                """
Try Nanonets-OCR-s<br>
We’ve open-sourced Nanonets-OCR-s, A model for transforming documents into structured markdown with content recognition and semantic tagging.<br>
📖 [Release Blog](https://huggingface.co/nanonets/Nanonets-OCR-s) 🤗 [View on Hugging Face](https://huggingface.co/nanonets/Nanonets-OCR-s)
""",
                visible=True,
            ) if model_name != "hosted_vllm/nanonets/Nanonets-OCR-s" else None

            file_input = gr.File(
                label="Upload Documents",
                file_types=[
                    ".pdf",
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".tiff",
                    ".bmp",
                    ".gif",
                    ".webp",
                ],
                file_count="multiple",
            )
            images_input = gr.Gallery(
                label="Document Preview", preview=True, visible=False
            )
            submit_btn = gr.Button("Submit", visible=False)

            def handle_file_upload(files):
                if not files:
                    return None, gr.update(visible=False), gr.update(visible=False)

                file_paths = [f.name for f in files]
                # Convert PDFs to images if necessary and get all image paths
                image_paths = convert_files_to_images(file_paths)
                return (
                    image_paths,
                    gr.update(visible=True, value=image_paths),
                    gr.update(visible=True),
                )

            file_input.change(
                handle_file_upload,
                inputs=[file_input],
                outputs=[images_input, images_input, submit_btn],
            )

            formatted_output = gr.Markdown(
                label="Formatted model prediction",
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {
                        "left": "\\begin{align*}",
                        "right": "\\end{align*}",
                        "display": True,
                    },
                ],
                line_breaks=True,
                show_copy_button=True,
            )

            # Renamed for clarity, this is the VLLM path using images
            def process_images_to_markdown_stream(images_list, request_id_prefix="vllm"):
                """
                Process images to markdown with streaming updates (page by page) using VLLM.
                """
                request_id = f"{request_id_prefix}-{str(uuid.uuid4())[:8]}"
                logger.info(f"[{request_id}] Starting VLLM processing for {len(images_list)} images.")

                num_pages = len(images_list) if images_list else 0
                current_page = 1
                try:
                    for markdown_content_chunk in convert_to_markdown_stream(
                        images_list, # expects a list of image paths
                        model_name,
                        max_img_size,
                        concurrency_limit, # This might need to be handled differently if Mistral part is also concurrent
                        max_gen_tokens,
                    ):
                        if num_pages > 1:
                            progress_header = f"📄 **VLLM Conversion Progress** `[Request {request_id}]` (Processing page {min(current_page, num_pages)} of {num_pages})\n\n"
                            yield progress_header + process_tags(markdown_content_chunk)
                        else:
                            yield process_tags(markdown_content_chunk)

                        if "---" in markdown_content_chunk: # A simple way to track pages
                            current_page = markdown_content_chunk.count("---") + 1
                        time.sleep(0.01)
                    logger.info(f"[{request_id}] VLLM processing completed.")
                except Exception as e:
                    logger.error(f"[{request_id}] VLLM processing failed: {e}")
                    error_message = f"❌ **Error in VLLM processing (Request {request_id})**: {str(e)}"
                    yield error_message

            def process_uploaded_files_to_markdown(files, progress=gr.Progress(track_tqdm=True)):
                if not files:
                    yield "No files uploaded. Please upload one or more documents."
                    return

                logger.info(f"Processing {len(files)} uploaded files.")
                mistral_api_key = os.getenv("MISTRAL_API_KEY")
                mistral_client = None

                if mistral_api_key:
                    try:
                        mistral_client = MistralOCRClient()
                        logger.info("MistralOCRClient initialized successfully.")
                    except Exception as e:
                        logger.warning(f"Failed to initialize MistralOCRClient (MISTRAL_API_KEY is set): {e}. Will fallback to VLLM.")
                        yield f"⚠️ **Warning**: Could not initialize Mistral OCR client: {e}. Falling back to default processing.\n\n---\n\n"
                        mistral_client = None # Ensure it's None so fallback occurs
                else:
                    logger.info("MISTRAL_API_KEY not found. Using VLLM for markdown conversion.")

                for file_obj in progress.tqdm(files, desc="Processing documents"):
                    file_path = file_obj.name
                    request_id = str(uuid.uuid4())[:8]
                    logger.info(f"[{request_id}] Processing file: {file_path}")

                    use_vllm_fallback = True
                    if mistral_client:
                        try:
                            logger.info(f"[{request_id}] Attempting Mistral OCR for {file_path}...")
                            yield f"⏳ Attempting Mistral OCR for {os.path.basename(file_path)}...\n\n---\n\n"
                            # progress.update(f"Mistral OCR: {os.path.basename(file_path)}")
                            markdown_output = mistral_client.extract_text_from_file(file_path)
                            logger.info(f"[{request_id}] Mistral OCR successful for {file_path}.")

                            # Mistral client returns markdown with pages separated by "\n\n---\n\n"
                            # We can yield page by page to somewhat match the streaming behavior
                            pages = markdown_output.split("\n\n---\n\n")
                            num_pages = len(pages)
                            for i, page_md in enumerate(pages):
                                if num_pages > 1:
                                    page_header = f"📄 **Mistral OCR Output** `[File {os.path.basename(file_path)}]` (Page {i+1} of {num_pages})\n\n"
                                    yield page_header + process_tags(page_md) + "\n\n---\n\n"
                                else:
                                    yield process_tags(page_md)
                                time.sleep(0.01) # Small delay for UI update
                            use_vllm_fallback = False
                        except Exception as e:
                            logger.error(f"[{request_id}] Mistral OCR failed for {file_path}: {e}")
                            yield f"⚠️ **Mistral OCR Error for {os.path.basename(file_path)}**: {e}. Falling back to VLLM.\n\n---\n\n"
                            # Fall through to VLLM if Mistral fails for this file

                    if use_vllm_fallback:
                        logger.info(f"[{request_id}] Using VLLM fallback for {file_path}.")
                        yield f"⏳ Using default VLLM model for {os.path.basename(file_path)} (converting to images first)...\n\n---\n\n"
                        # progress.update(f"VLLM (converting): {os.path.basename(file_path)}")
                        try:
                            # 1. Convert the individual file to image(s)
                            image_paths = convert_files_to_images([file_path], max_img_size=max_img_size)
                            if not image_paths:
                                logger.warning(f"[{request_id}] Could not convert {file_path} to images for VLLM.")
                                yield f"⚠️ Could not convert {os.path.basename(file_path)} to images for VLLM processing.\n\n---\n\n"
                                continue # Skip to next file

                            logger.info(f"[{request_id}] Converted {file_path} to {len(image_paths)} image(s) for VLLM.")
                            # progress.update(f"VLLM (processing): {os.path.basename(file_path)}")
                            # 2. Process these images using the existing VLLM streaming logic
                            for vllm_output_chunk in process_images_to_markdown_stream(image_paths, request_id_prefix=f"{request_id}-vllm"):
                                yield vllm_output_chunk
                        except Exception as e:
                            logger.error(f"[{request_id}] VLLM fallback processing for {file_path} failed: {e}")
                            yield f"❌ **Error in VLLM fallback for {os.path.basename(file_path)}**: {e}\n\n---\n\n"

                    yield f"✅ Finished processing {os.path.basename(file_path)}.\n\n*****\n\n"


            submit_btn.click(
                process_uploaded_files_to_markdown, # New function
                inputs=[file_input], # Changed from images_input
                outputs=[formatted_output],
                concurrency_limit=concurrency_limit,
                concurrency_id="pdf_to_markdown_conversion",
            )
