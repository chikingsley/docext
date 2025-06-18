# Mistral Document AI OCR - Python Guide

## Overview

Mistral Document AI OCR processor powered by the `mistral-ocr-latest` model enables you to extract text and structured content from PDF documents and images with high accuracy.

## Key Features

- **Structure Preservation**: Extracts text while maintaining document structure and hierarchy
- **Formatting Support**: Preserves headers, paragraphs, lists, and tables
- **Markdown Output**: Returns results in markdown format for easy parsing and rendering
- **Complex Layouts**: Handles multi-column text and mixed content
- **Scalable Processing**: Processes documents at scale with high accuracy

## Supported Formats

- **Images**: PNG, JPEG/JPG, AVIF, and more
- **Documents**: PDF, PPTX, DOCX, and more

## Setup

```python
import os
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)
```

## OCR with PDF Documents

### Method 1: Direct PDF URL

```python
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": "https://arxiv.org/pdf/2201.04234"
    },
    include_image_base64=True
)
```

### Method 2: Base64 Encoded PDF

```python
import base64

def encode_pdf(pdf_path):
    """Encode the pdf to base64."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Path to your pdf
pdf_path = "path_to_your_pdf.pdf"

# Getting the base64 string
base64_pdf = encode_pdf(pdf_path)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}" 
    },
    include_image_base64=True
)
```

### Method 3: Upload PDF File

#### Step 1: Upload the file

```python
uploaded_pdf = client.files.upload(
    file={
        "file_name": "uploaded_file.pdf",
        "content": open("uploaded_file.pdf", "rb"),
    },
    purpose="ocr"
)
```

#### Step 2: Retrieve file information

```python
retrieved_file = client.files.retrieve(file_id=uploaded_pdf.id)
print(retrieved_file)
# Output: id='00edaf84-95b0-45db-8f83-f71138491f23' object='file' size_bytes=3749788 created_at=1741023462 filename='uploaded_file.pdf' purpose='ocr' sample_type='ocr_input' source='upload' deleted=False num_lines=None
```

#### Step 3: Get signed URL

```python
signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
```

#### Step 4: Process OCR

```python
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": signed_url.url,
    },
    include_image_base64=True
)
```

## OCR with Images

### Method 1: Direct Image URL

```python
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "image_url",
        "image_url": "https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/receipt.png"
    },
    include_image_base64=True
)
```

### Method 2: Base64 Encoded Image

```python
import base64

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Path to your image
image_path = "path_to_your_image.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{base64_image}" 
    },
    include_image_base64=True
)
```

## Response Structure

The OCR processor returns a JSON response with the following structure:

```json
{
    "pages": [
        {
            "index": 1,
            "markdown": "# LEVERAGING UNLABELED DATA TO PREDICT OUT-OF-DISTRIBUTION PERFORMANCE\n\nSaurabh Garg*<br>Carnegie Mellon University<br>sgarg2@andrew.cmu.edu<br>Sivaraman Balakrishnan<br>Carnegie Mellon University<br>sbalakri@andrew.cmu.edu\n\n## Abstract\n\nReal-world machine learning deployments are characterized by mismatches between the source (training) and target (test) distributions that may cause performance drops. In this work, we investigate methods for predicting the target domain accuracy using only labeled source data and unlabeled target data.",
            "images": [],
            "dimensions": {
                "dpi": 200,
                "height": 2200,
                "width": 1700
            }
        },
        {
            "index": 2,
            "markdown": "![img-0.jpeg](img-0.jpeg)\n\nFigure 1: Illustration of our proposed method ATC. Left: using source domain validation data, we identify a threshold on a score (e.g. negative entropy) computed on model confidence such that fraction of examples above the threshold matches the validation set accuracy.\n\nRecently, numerous methods have been proposed for this purpose (Deng & Zheng, 2021; Chen et al., 2021b; Jiang et al., 2021). These methods either require calibration on the target domain to yield consistent estimates or additional labeled data from several target domains.",
            "images": [
                {
                    "id": "img-0.jpeg",
                    "top_left_x": 292,
                    "top_left_y": 217,
                    "bottom_right_x": 1405,
                    "bottom_right_y": 649,
                    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                }
            ],
            "dimensions": {
                "dpi": 200,
                "height": 2200,
                "width": 1700
            }
        },
        {
            "index": 3,
            "markdown": "| Dataset | Shift | IM |  | AC |  | DOC |  | GDE | ATC-MC (Ours) |  | ATC-NE (Ours) |  |\n| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |\n|  |  | Pre T | Post T | Pre T | Post T | Pre T | Post T | Post T | Pre T | Post T | Pre T | Post T |\n| CIFAR10 | Natural | 6.60 | 5.74 | 9.88 | 6.89 | 7.25 | 6.07 | 4.77 | 3.21 | 3.02 | 2.99 | 2.85 |\n| CIFAR100 | Synthetic | 13.69 | 11.51 | 23.61 | 13.10 | 14.60 | 10.14 | 9.85 | 5.50 | 4.75 | 4.72 | 4.94 |\n\nTable 1: Mean Absolute estimation Error (MAE) results for different datasets grouped by shift type.\n\n## Mathematical Formulations\n\nOur method identifies a threshold $t$ on source data $\\mathcal{D}^{\\mathbb{S}}$ such that:\n\n$\n\\mathbb{E}_{x \\sim \\mathcal{D}^{\\mathbb{S}}}[\\mathbb{I}[s(f(x))<t]]=\\mathbb{E}_{(x, y) \\sim \\mathcal{D}^{\\mathbb{S}}}\\left[\\mathbb{I}\\left[\\arg \\max _{j \\in \\mathcal{Y}} f_{j}(x) \\neq y\\right]\\right]\n$\n\n### Key Features:\n\n- Extracts text content while maintaining document structure\n- Preserves formatting like **headers**, *paragraphs*, and lists  \n- Returns results in markdown format for easy parsing\n- Handles complex layouts including multi-column text",
            "images": [],
            "dimensions": {
                "dpi": 200,
                "height": 2200,
                "width": 1700
            }
        }
    ],
    "model": "mistral-ocr-latest",
    "usage_info": {
        "pages_processed": 3,
        "doc_size_bytes": 1024000
    }
}
```

### Response Components

- **`pages`**: Array of page objects containing:
  - **`index`**: Page number (1-based indexing)
  - **`markdown`**: Extracted text content in markdown format including:
    - **Headers**: `# Title`, `## Subtitle`, `### Section`
    - **Text formatting**: `**bold**`, `*italic*`, regular text
    - **Images**: `![img-0.jpeg](img-0.jpeg)` syntax for embedded images
    - **Tables**: Full markdown table syntax with alignment
    - **Mathematical equations**: LaTeX-style formulas with `$` blocks
    - **Lists**: Both bulleted and numbered lists
    - **Links and citations**: Preserved reference formatting
    - **Line breaks**: `<br>` tags and paragraph separation
  - **`images`**: Array of detected images with:
    - **`id`**: Image filename reference matching markdown syntax
    - **`top_left_x/y`**: Bounding box coordinates (top-left corner)
    - **`bottom_right_x/y`**: Bounding box coordinates (bottom-right corner)  
    - **`image_base64`**: Base64-encoded image data (when `include_image_base64=True`)
  - **`dimensions`**: Page dimensions including:
    - **`dpi`**: Dots per inch resolution
    - **`height`**: Page height in pixels
    - **`width`**: Page width in pixels
- **`model`**: The OCR model used for processing (`mistral-ocr-latest`)
- **`usage_info`**: Processing metadata including:
  - **`pages_processed`**: Total number of pages processed
  - **`doc_size_bytes`**: Original document size in bytes

## API Limitations

- **File Size**: Maximum 50 MB per document
- **Page Limit**: Maximum 1,000 pages per document

## Additional Resources

For more advanced usage and examples, check out Mistral's cookbooks:
- Tool Use
- Batch OCR

## Best Practices

1. **Error Handling**: Always implement proper error handling when encoding files
2. **File Management**: Clean up uploaded files when no longer needed
3. **Format Selection**: Choose the appropriate input method based on your use case:
   - Direct URLs for publicly accessible documents
   - Base64 encoding for local files
   - File upload for better file management and reuse