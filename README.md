<!-- ---
license: mit
language:
  - vi
  - en
tags:
  - computer-vision
  - document-segmentation
  - document-scanner
  - ocr
  - fastapi
  - opencv
  - yolo
  - mineru
pipeline_tag: image-segmentation
--- -->

# Document Segmentation FastAPI

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://fastapi.tiangolo.com/"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi&logoColor=white"></a>
  <a href="https://opencv.org/"><img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8?style=flat-square&logo=opencv&logoColor=white"></a>
  <a href="https://github.com/ultralytics/ultralytics"><img alt="YOLO" src="https://img.shields.io/badge/YOLO-Segmentation-111827?style=flat-square"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-16A34A?style=flat-square"></a>
</p>

<p align="center">
  <b>Web app for document segmentation, perspective correction, scan enhancement, and OCR.</b>
</p>

<p align="center">
  <img src="static/assets/image.png" alt="Document Segmentation FastAPI interface preview" width="860">
</p>

<p align="center">
  <sub>FastAPI + OpenCV + YOLO Segmentation + MinerU OCR</sub>
</p>

## Overview

Document Segmentation FastAPI is a web-based document scanning demo. It detects the document region from uploaded images, estimates or accepts four document corners, applies perspective correction, enhances the final scan, and optionally runs OCR on the processed result.

The project is designed for computer vision coursework, document scanner prototyping, YOLO segmentation experiments, and OCR post-processing workflows.

## Highlights

- Multiple processing modes: OpenCV pipeline, YOLO segmentation, Hough Transform, and Manual Contour.
- Batch image upload with queue status, progress tracking, intermediate outputs, and final scan preview.
- Configurable preprocessing, edge detection, morphology, contour filtering, corner detection, and enhancement parameters.
- YOLO `.pt` upload and reuse for document segmentation masks.
- Manual four-corner selection for difficult images.
- MinerU OCR on the final scanned image, with Markdown, JSON, PDF, and visual output links when available.
- Responsive HTML/CSS/JavaScript interface for desktop and mobile.

## Tech Stack

| Layer | Stack |
| --- | --- |
| Backend | FastAPI, Uvicorn, Pydantic |
| Computer Vision | OpenCV, NumPy |
| Segmentation | Ultralytics YOLO |
| OCR | MinerU |
| Frontend | HTML, CSS, JavaScript |
| Storage | Local filesystem: `uploads/`, `outputs/`, `models/` |

## Processing Modes

| Mode | Description | Best For |
| --- | --- | --- |
| OpenCV Pipeline | Traditional preprocessing, edge detection, morphology, contour detection, and perspective transform. | Clear document boundaries and parameter tuning. |
| YOLO Segmentation | Uses a trained YOLO segmentation model to extract a document mask before scanning. | Complex backgrounds or trained production-like segmentation. |
| Hough Transform | Detects document boundary lines with HoughLinesP and reconstructs the main contour. | Documents with strong straight edges. |
| Manual Contour | Lets the user click four document corners manually. | Hard cases where automatic detection fails. |

## Pipeline

```text
Upload image
  -> Select processor: OpenCV / YOLO / Hough / Manual
  -> Detect document region
  -> Estimate or receive 4 corners
  -> Perspective transform
  -> Image enhancement
  -> Final scanned output
  -> Optional MinerU OCR
```

## Quickstart

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Optional MinerU OCR installation:

```bash
uv pip install -U "mineru[all]"
```

## Usage

### OpenCV Pipeline

1. Upload one or more document images.
2. Select `Pipeline gốc`.
3. Adjust image mode, illumination correction, edge detection, morphology, and detailed parameters if needed.
4. Enable or disable pipeline steps.
5. Click `Chạy xử lý`.

### YOLO Segmentation

1. Upload one or more document images.
2. Select `YOLO Segment`.
3. Upload a trained YOLO `.pt` segmentation model.
4. Tune `YOLO confidence` and `Mask threshold`.
5. Click `Chạy xử lý`.

YOLO flow:

```text
Original image -> YOLO mask -> largest contour -> 4 corners -> perspective transform -> enhancement
```

If YOLO returns masks, the app uses the mask to find the document contour. If only boxes are available, it falls back to the largest box.

### Hough Transform

1. Upload a document image.
2. Select `Hough Transform`.
3. Tune Canny, Hough threshold, minimum line length, maximum line gap, and morphology settings.
4. Click `Chạy xử lý`.

### Manual Contour

1. Upload a document image.
2. Select `Manual Contour`.
3. Click `Chọn góc` and select four document corners on the preview.
4. Adjust enhancement if needed.
5. Click `Chạy xử lý`.

### MinerU OCR

1. Run any scan pipeline first.
2. In the result card, click `OCR` next to the final image.
3. The server runs MinerU on the scanned output.
4. View Markdown preview and download generated OCR artifacts.

OCR results are stored in:

```text
outputs/<image_id>/ocr/
```

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/upload` | Upload one or more images with form key `files`. |
| `POST` | `/api/model/upload` | Upload a YOLO `.pt` model with form key `file`. |
| `GET` | `/api/model/status` | Check the active YOLO model. |
| `POST` | `/api/run` | Run the selected document processing pipeline. |
| `POST` | `/api/ocr/{image_id}` | Run MinerU OCR on a processed result image. |
| `GET` | `/api/status` | Get queue and processing status. |
| `GET` | `/api/results/{image_id}` | Get outputs for one image. |
| `GET` | `/api/download/{image_id}/{file_path}` | Download an output artifact. |
| `DELETE` | `/api/clear` | Clear uploaded images and generated outputs. |
| `DELETE` | `/api/cache` | Alias for clearing cache. |

Example OpenCV request:

```json
{
  "config": {
    "processor": "opencv",
    "mode": "color",
    "illumination_method": "lab",
    "gray_equalization_method": "clahe",
    "edge_method": "canny",
    "morph_operation": "closing",
    "steps": {
      "gaussian_blur": true,
      "median_blur": true,
      "sharpen": true,
      "illumination": true,
      "edge_detection": true,
      "morphology": true,
      "find_contour": true,
      "detect_corners": true,
      "perspective_transform": true,
      "enhance": true
    },
    "params": {
      "gaussian_ksize": 5,
      "median_ksize": 5,
      "sharpen_amount": 1.0,
      "clahe_clip_limit": 2.0,
      "clahe_tile_grid": 8,
      "sobel_ksize": 3,
      "sobel_threshold": 60,
      "canny_low": 50,
      "canny_high": 150,
      "morph_kernel": 5,
      "morph_iterations": 1,
      "contour_min_area_ratio": 0.08,
      "enhance_alpha": 1.15,
      "enhance_beta": 8
    }
  }
}
```

Example YOLO request:

```json
{
  "config": {
    "processor": "yolo",
    "params": {
      "yolo_confidence": 0.25,
      "yolo_mask_threshold": 0.5,
      "enhance_alpha": 1.15,
      "enhance_beta": 8
    },
    "yolo": {
      "model_path": ""
    }
  }
}
```

`model_path` can stay empty because the backend automatically uses the latest uploaded `.pt` model.

Example OCR request:

```json
{
  "config": {
    "mineru": {
      "method": "ocr",
      "backend": "pipeline",
      "lang": "",
      "formula": true,
      "table": true,
      "timeout_seconds": 3600
    }
  }
}
```

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── ocr.py
│   ├── segmentation.py
│   └── storage.py
├── static/
│   ├── assets/
│   │   ├── Example.png
│   │   └── image.png
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
├── main.py
├── requirements.txt
├── script.sh
├── LICENSE
└── README.md
```

Runtime directories are created automatically:

- `uploads/`: uploaded source images.
- `outputs/`: processed images and OCR artifacts.
- `models/`: uploaded YOLO `.pt` models.

## Configuration Notes

| Parameter | Purpose |
| --- | --- |
| `canny_low`, `canny_high` | Canny edge thresholds for the OpenCV pipeline. |
| `morph_kernel`, `morph_iterations` | Morphology kernel size and iteration count. |
| `contour_min_area_ratio` | Minimum contour area relative to image size. |
| `yolo_confidence` | YOLO confidence threshold. |
| `yolo_mask_threshold` | Binary threshold for YOLO masks. |
| `enhance_alpha`, `enhance_beta` | Contrast and brightness enhancement. |
| `timeout_seconds` | MinerU OCR timeout. |

## Limitations

- YOLO quality depends on the uploaded segmentation model.
- OCR runtime can be slow for dense documents, formulas, and tables.
- Uploaded files and generated outputs are stored locally, so long-running deployments should clean cache regularly.
- This repository is a demo/prototype, not a hardened multi-user production service.

## Roadmap

- Multi-model YOLO management.
- Batch ZIP export.
- Dockerfile and docker-compose setup.
- Automated API and image-processing tests.

## License

Released under the [MIT License](LICENSE).
