# Document Segmentation FastAPI

Ứng dụng web demo phân đoạn tài liệu bằng **FastAPI + OpenCV** và có thêm chế độ **YOLO segmentation**. Pipeline gốc vẫn được giữ nguyên để có thể bật/tắt từng bước và chỉnh tham số trực tiếp trên giao diện.

![Scan app](https://raw.githubusercontent.com/ntrThanh/document-scanner/refs/heads/master/static/assets/Example.png)

## Chức năng chính

- Upload một hoặc nhiều ảnh.
- Chọn 1 trong 2 chế độ xử lý:
  - **Pipeline gốc OpenCV**: Gaussian, Median, Sharpen, cân bằng sáng, phát hiện biên, morphology, contour, phát hiện góc, perspective transform, enhancement.
  - **YOLO Segment**: upload file `.pt`, chạy YOLO segmentation, lấy mask/contour, phát hiện 4 góc, chuyển đổi góc nhìn, tăng cường ảnh.
- Giao diện responsive, dùng được trên máy tính và điện thoại.
- Xem hàng chờ xử lý, tiến trình, ảnh gốc, ảnh cuối và ảnh trung gian.
- Tải xuống từng ảnh kết quả.

## Cấu trúc project

```text
document_segmentation_fastapi/
├── app/
│   ├── __init__.py
│   ├── segmentation.py
│   └── storage.py
├── static/
│   ├── style.css
│   └── script.js
├── templates/
│   └── index.html
├── uploads/
├── outputs/
├── models/
├── main.py
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu chỉ chạy pipeline OpenCV thì không cần model YOLO. Nếu chạy YOLO, cần có file model `.pt` đã train cho bài toán segmentation tài liệu.

## Chạy ứng dụng

```bash
uvicorn main:app --reload
```

Mở trình duyệt:

```text
http://127.0.0.1:8000
```

## Cách dùng trên giao diện

### 1. Pipeline gốc

1. Upload ảnh.
2. Chọn **Pipeline gốc**.
3. Chỉnh kiểu ảnh, phương pháp cân bằng sáng, phát hiện biên, morphology.
4. Bật/tắt các bước xử lý nếu cần.
5. Mở phần **Tham số pipeline gốc** để chỉnh Canny, Sobel, CLAHE, kernel, alpha/beta...
6. Bấm **Chạy xử lý**.

### 2. YOLO Segment

1. Upload ảnh.
2. Chọn **YOLO Segment**.
3. Upload file model `.pt`.
4. Chỉnh `YOLO confidence` và `Mask threshold` nếu cần.
5. Bấm **Chạy xử lý**.

Luồng YOLO:

```text
Ảnh gốc
→ YOLO mask
→ lấy contour lớn nhất
→ phát hiện 4 góc
→ perspective transform
→ tăng cường ảnh
```

Nếu model YOLO trả về mask, hệ thống dùng mask để tìm contour. Nếu không có mask nhưng có box, hệ thống fallback sang box lớn nhất để ước lượng vùng tài liệu.

## API

### Upload ảnh

```http
POST /api/upload
```

Body dạng `multipart/form-data`, key là `files`.

### Upload model YOLO

```http
POST /api/model/upload
```

Body dạng `multipart/form-data`, key là `file`, chỉ nhận `.pt`.

### Kiểm tra model YOLO hiện tại

```http
GET /api/model/status
```

### Chạy xử lý

```http
POST /api/run
```

Ví dụ chạy pipeline gốc:

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

Ví dụ chạy YOLO:

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

`model_path` không cần truyền từ frontend vì server tự dùng model `.pt` đã upload gần nhất.

### Lấy trạng thái

```http
GET /api/status
```

### Lấy kết quả một ảnh

```http
GET /api/results/{image_id}
```

### Tải ảnh kết quả

```http
GET /api/download/{image_id}/{filename}
```

### Xóa hàng chờ

```http
DELETE /api/clear
```

## Ghi chú

- Kết quả được lưu trong thư mục `outputs/`.
- Ảnh upload được lưu trong thư mục `uploads/`.
- Model YOLO upload được lưu trong thư mục `models/`.
- Pipeline OpenCV phù hợp để demo học thuật và cho phép tinh chỉnh chi tiết.
- Chế độ YOLO phụ thuộc nhiều vào chất lượng model `.pt`; nên train model YOLO segmentation với nhãn mask/segment tài liệu.
