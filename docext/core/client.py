from __future__ import annotations

import os
import base64
from mistralai import Mistral
from mistralai.models.ocr import OCRResponse
import mimetypes
from dotenv import load_dotenv # Added

import requests
from litellm import completion

load_dotenv() # Load environment variables from .env file

def sync_request(
    messages: list[dict],
    model_name: str = "hosted_vllm/Qwen/Qwen2.5-VL-3B-Instruct",
    max_tokens: int = 5000,
    num_completions: int = 1,
    format: dict | None = None,
):
    vlm_url = os.getenv("VLM_MODEL_URL", "")
    if vlm_url == "":
        raise ValueError(
            "VLM_MODEL_URL is not set. Please set it to the URL of the VLM model.",
        )
    completion_args = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "n": num_completions,
        "temperature": 0,
        "api_base": vlm_url
        if model_name.startswith("hosted_vllm/") or model_name.startswith("ollama/")
        else None,
    }

    if model_name.startswith("hosted_vllm/") or model_name.startswith("ollama/"):
        completion_args["api_key"] = os.getenv("API_KEY", "EMPTY")

    # Only add format argument for Ollama models
    if model_name.startswith("ollama/") and format:
        completion_args["format"] = format
    # elif model_name.startswith("hosted_vllm/") and format: # TODO: Add this back, currently not working in colab
    #     completion_args["guided_json"] = format
    #     if "qwen" in model_name.lower():
    #         completion_args["guided_backend"] = "xgrammar:disable-any-whitespace"
    elif model_name.startswith("openrouter"):
        completion_args["response_format"] = format
    elif "gpt" in model_name.lower():
        # Only set response_format if the prompt mentions "json"
        if any("json" in m.get("text", "").lower() for m in messages if isinstance(m, dict)):
            completion_args["response_format"] = {"type": "json_object"}

    response = completion(**completion_args)
    return response.json()


class MistralOCRClient:
    def __init__(self):
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set (or not found in .env file).")
        self.client = Mistral(api_key=self.api_key)

    def _encode_file_to_base64(self, file_path: str) -> str:
        try:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode('utf-8')
        except FileNotFoundError:
            # Log or raise a more specific error
            print(f"Error: The file {file_path} was not found.")
            raise
        except Exception as e:
            # Log or raise
            print(f"Error encoding file: {e}")
            raise

    def extract_text_from_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        base64_encoded_file = self._encode_file_to_base64(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)

        if not mime_type:
            # Attempt to infer from extension if mimetypes fails
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                mime_type = "application/pdf"
            elif ext in [".png", ".jpg", ".jpeg", ".avif", ".webp"]: # Add other supported image types
                mime_type = f"image/{ext[1:]}" # e.g. image/png
            else:
                raise ValueError(f"Could not determine MIME type for file: {file_path}")

        document_type_prefix = ""
        if mime_type == "application/pdf":
            document_type_prefix = "data:application/pdf;base64,"
            doc_input_type = "document_url" # Mistral uses 'document_url' for base64 encoded PDFs too
        elif mime_type.startswith("image/"):
            document_type_prefix = f"data:{mime_type};base64,"
            doc_input_type = "image_url" # Mistral uses 'image_url' for base64 encoded images
        else:
            raise ValueError(f"Unsupported MIME type: {mime_type} for file: {file_path}")

        try:
            ocr_response: OCRResponse = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": doc_input_type,
                    # The API expects a URL-like format even for base64 data
                    doc_input_type.split('_')[0] + "_url": f"{document_type_prefix}{base64_encoded_file}"
                },
                include_image_base64=False # Not needed for text extraction
            )

            all_markdown = []
            for page in ocr_response.pages:
                all_markdown.append(page.markdown)

            return "\n\n---\n\n".join(all_markdown) # Join pages with a separator

        except Exception as e:
            # Log error, perhaps raise a custom exception
            print(f"Mistral OCR API request failed: {e}")
            # Consider how to handle partial success if some pages process but others fail
            raise
