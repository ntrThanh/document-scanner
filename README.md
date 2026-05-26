# Scan Tài Liệu - Document Segmentation FastAPI

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://fastapi.tiangolo.com/"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi&logoColor=white"></a>
  <a href="https://opencv.org/"><img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8?style=flat-square&logo=opencv&logoColor=white"></a>
  <a href="https://github.com/ultralytics/ultralytics"><img alt="YOLO" src="https://img.shields.io/badge/YOLO-Segmentation-111827?style=flat-square"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-16A34A?style=flat-square"></a>
</p>

<p align="center">
  <b>Ứng dụng web scan tài liệu: phát hiện vùng giấy, cắt phối cảnh, tăng cường ảnh và OCR.</b>
</p>

<p align="center">
  <a href="static/assets/README_EN.md">English version</a>
</p>

<p align="center">
  <img src="static/assets/image.png" alt="Giao diện Scan tài liệu" width="860">
</p>

## Giới Thiệu

Project này là một demo scan tài liệu bằng FastAPI, OpenCV và YOLO Segmentation. Người dùng có thể upload ảnh tài liệu, chọn chế độ xử lý, chạy pipeline để lấy ảnh scan cuối cùng, sau đó OCR bằng MinerU nếu cần.

Ứng dụng phù hợp cho bài tập Computer Vision, thử nghiệm document scanner, đánh giá mô hình YOLO segmentation và xử lý hậu kỳ OCR.

## Tính Năng

- Upload nhiều ảnh và xử lý theo hàng chờ.
- Scan nhanh bằng YOLO với preset có sẵn.
- Chế độ YOLO tùy chỉnh với model `.pt`.
- Pipeline OpenCV truyền thống để chỉnh từng bước xử lý.
- Hough Transform để dò biên tài liệu theo đường thẳng.
- Manual Contour để chọn 4 góc thủ công khi ảnh khó.
- Tăng cường ảnh sau khi cắt bằng contrast, Otsu hoặc Adaptive Threshold.
- OCR bằng MinerU và xem/tải các file kết quả như Markdown, JSON, PDF, ảnh visualize.
- Lưu ảnh upload, output và model YOLO trên filesystem local.

## Công Nghệ Sử Dụng

| Phần | Công nghệ |
| --- | --- |
| Backend | FastAPI, Uvicorn, Pydantic |
| Computer Vision | OpenCV, NumPy, Pillow |
| Segmentation | Ultralytics YOLO |
| OCR | MinerU |
| Frontend | HTML, CSS, JavaScript |
| Lưu trữ | `uploads/`, `outputs/`, `models/` |

## Cài Đặt

Yêu cầu Python 3.10 trở lên.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu cần OCR, cài thêm MinerU:

```bash
uv pip install -U "mineru[all]"
```

## Chạy Ứng Dụng

Chạy dev bằng Uvicorn có reload:

```bash
bash script.sh dev
```

Mở trình duyệt tại:

```text
http://127.0.0.1:8888
```

Chạy production bằng Gunicorn + Uvicorn worker:

```bash
bash script.sh
```

Server mặc định chạy tại:

```text
http://0.0.0.0:8888
```

### Cấu Hình Chạy Nhiều Người Dùng

Backend đã có hàng đợi xử lý ảnh song song trong cùng một process. Chỉnh số job chạy đồng thời bằng biến môi trường:

```bash
SCAN_MAX_CONCURRENT_JOBS=2 SCAN_MAX_CONCURRENT_OCR=1 bash script.sh
```

Gợi ý cấu hình:

| Máy chạy | `SCAN_MAX_CONCURRENT_JOBS` | `SCAN_MAX_CONCURRENT_OCR` |
| --- | ---: | ---: |
| CPU laptop/PC thường | 1-2 | 1 |
| CPU nhiều core | 2-4 | 1-2 |
| GPU dùng YOLO/MinerU | 1-2 | 1 |

Project hiện lưu trạng thái job trong RAM, vì vậy production config mặc định dùng `WEB_CONCURRENCY=1`. Không tăng nhiều Gunicorn worker nếu chưa chuyển `JobStore` sang Redis/database, vì mỗi worker sẽ có hàng đợi riêng và frontend có thể poll nhầm worker.

## Cách Sử Dụng

### 1. Scan Nhanh Bằng YOLO

1. Upload ảnh tài liệu.
2. Chọn hoặc upload model YOLO `.pt`.
3. Bấm `Upload và scan` hoặc `Scan ảnh đã upload`.
4. Xem ảnh scan và các bước trung gian ở phần kết quả.

Chế độ này dùng preset mặc định cho tài liệu, gồm phát hiện mask bằng YOLO, cắt phối cảnh và tăng cường ảnh bằng threshold.

### 2. YOLO Tùy Chỉnh

1. Mở `Khám phá nâng cao`.
2. Chọn `YOLO tùy chỉnh`.
3. Upload/dùng lại model `.pt`.
4. Chỉnh confidence, mask threshold, preprocessing và kiểu tăng cường ảnh.
5. Chạy scan.

Nếu model trả về mask, app dùng mask để tìm contour tài liệu. Nếu không có mask, app fallback theo bounding box lớn nhất.

### 3. Pipeline OpenCV

1. Mở `Khám phá nâng cao`.
2. Chọn `Pipeline gốc`.
3. Chỉnh các bước như blur, sharpen, cân bằng sáng, edge detection, morphology, contour và perspective transform.
4. Chạy scan để xem từng output trung gian.

### 4. Hough Transform

1. Chọn `Hough Transform`.
2. Chỉnh Canny, Hough threshold, min line length, max line gap và morphology.
3. Chạy scan để dò cạnh tài liệu theo các đường thẳng.

### 5. Manual Contour

1. Chọn `Manual Contour`.
2. Bấm chọn góc và click 4 góc của tài liệu trên ảnh preview.
3. Chạy scan để cắt phối cảnh theo 4 điểm đã chọn.

### 6. OCR Bằng MinerU

1. Chạy scan xong một ảnh.
2. Bấm `OCR` trên result card.
3. Đợi MinerU xử lý.
4. Xem Markdown preview hoặc tải artifact được sinh ra.

Kết quả OCR nằm trong:

```text
outputs/<image_id>/ocr/
```

## API Chính

| Method | Endpoint | Chức năng |
| --- | --- | --- |
| `GET` | `/` | Giao diện web. |
| `GET` | `/api/config` | Lấy config mặc định. |
| `POST` | `/api/upload` | Upload một hoặc nhiều ảnh, form key là `files`. |
| `POST` | `/api/model/upload` | Upload model YOLO `.pt`, form key là `file`. |
| `GET` | `/api/model/status` | Kiểm tra model YOLO đang dùng. |
| `POST` | `/api/run` | Chạy pipeline scan. |
| `GET` | `/api/status` | Lấy trạng thái hàng chờ. |
| `GET` | `/api/results/{image_id}` | Lấy kết quả xử lý của một ảnh. |
| `POST` | `/api/ocr/{image_id}` | OCR ảnh kết quả bằng MinerU. |
| `GET` | `/api/download/{image_id}/{file_path}` | Tải file output. |
| `DELETE` | `/api/clear` | Xóa ảnh upload và output, giữ lại model YOLO. |
| `DELETE` | `/api/cache` | Alias của `/api/clear`. |

Ví dụ request chạy YOLO:

```json
{
  "config": {
    "processor": "yolo",
    "params": {
      "yolo_confidence": 0.25,
      "yolo_mask_threshold": 0.5,
      "yolo_enhance_method": "otsu"
    }
  }
}
```

Ví dụ request OCR:

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

## Cấu Trúc Project

```text
.
├── app/
│   ├── __init__.py
│   ├── ocr.py
│   ├── segmentation.py
│   └── storage.py
├── models/
│   └── yolo_segmentation.pt
├── static/
│   ├── assets/
│   │   ├── Example.png
│   │   ├── README_EN.md
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

Các thư mục runtime:

- `uploads/`: ảnh gốc người dùng upload.
- `outputs/`: ảnh trung gian, ảnh scan cuối và kết quả OCR.
- `models/`: model YOLO `.pt` đã upload hoặc có sẵn.

## Hạn Chế

- Chế độ YOLO cần model segmentation `.pt`; nếu model train chưa tốt hoặc ảnh khác domain train, kết quả cắt tài liệu có thể sai.
- Pipeline OpenCV phụ thuộc nhiều vào ánh sáng, nền ảnh, độ tương phản và việc tài liệu có viền rõ hay không.
- Hough Transform hoạt động tốt hơn với tài liệu có cạnh thẳng, nhưng dễ lỗi khi ảnh bị cong, nhăn, lóa sáng hoặc nền có nhiều đường thẳng.
- Manual Contour yêu cầu người dùng chọn đúng 4 góc; nếu chọn lệch, ảnh sau khi cắt phối cảnh cũng sẽ lệch.
- MinerU là dependency tùy chọn, chỉ cần cài khi dùng OCR. Nếu máy chưa có lệnh `mineru`, chức năng OCR sẽ báo lỗi.
- OCR có thể chạy lâu với tài liệu nhiều bảng, công thức, ảnh lớn hoặc khi chạy trên máy không có GPU.
- Dữ liệu upload và output được lưu local trong `uploads/` và `outputs/`; nếu chạy lâu cần xóa cache định kỳ để tránh đầy ổ đĩa.
- Hệ thống dùng hàng đợi trong memory và có thể xử lý nhiều ảnh/request đồng thời trong một process, nhưng chưa phù hợp để scale nhiều process hoặc nhiều máy nếu chưa chuyển trạng thái job sang Redis/database.
- Project chưa có cơ chế đăng nhập, phân quyền, giới hạn dung lượng upload hoặc tự động dọn file theo thời gian.
- Đây là demo/prototype phục vụ học tập và thử nghiệm, chưa phải service production.

## License

Phát hành theo giấy phép [MIT](LICENSE).
