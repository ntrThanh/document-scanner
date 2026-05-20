import asyncio
import mimetypes
import os
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel

from app.ocr import MinerURunner
from app.segmentation import DEFAULT_CONFIG, SegmentationRunner
from app.storage import JobStore, clear_directory, safe_remove_path, save_upload_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI(title="Document Segmentation Demo")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
store = JobStore()
runner = SegmentationRunner(OUTPUT_DIR)
ocr_runner = MinerURunner(OUTPUT_DIR)
active_yolo_model = {"filename": None, "path": None}


class RunRequest(BaseModel):
    image_ids: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None


class OcrRequest(BaseModel):
    source_filename: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


def _latest_yolo_model():
    models = []
    for name in os.listdir(MODEL_DIR):
        if name.lower().endswith(".pt"):
            path = os.path.join(MODEL_DIR, name)
            if os.path.isfile(path):
                models.append((os.path.getmtime(path), name, path))
    if not models:
        return None
    _, name, path = max(models, key=lambda item: item[0])
    return {"filename": name, "path": path}


def _load_existing_yolo_model():
    model = _latest_yolo_model()
    if model:
        active_yolo_model.update(model)


def _safe_model_name(filename: str):
    safe_name = os.path.basename(filename or "model.pt").strip() or "model.pt"
    safe_name = safe_name.replace("/", "_").replace("\\", "_")
    if not safe_name.lower().endswith(".pt"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file model YOLO định dạng .pt")
    return safe_name


async def _save_or_reuse_yolo_model(file: UploadFile):
    safe_name = _safe_model_name(file.filename or "model.pt")
    path = os.path.join(MODEL_DIR, safe_name)

    # Nếu model cùng tên đã tồn tại thì dùng lại, không cần ghi/upload lại.
    if os.path.exists(path) and os.path.getsize(path) > 0:
        active_yolo_model["filename"] = safe_name
        active_yolo_model["path"] = path
        return safe_name, path, True

    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    active_yolo_model["filename"] = safe_name
    active_yolo_model["path"] = path
    return safe_name, path, False


def _delete_image_files(item: Dict[str, Any]):
    safe_remove_path(item.get("path"))
    safe_remove_path(os.path.join(OUTPUT_DIR, item.get("id", "")))


def _safe_output_path(image_id: str, file_path: str):
    safe_path = os.path.normpath(file_path or "").lstrip(os.sep)
    if not safe_path or safe_path.startswith(".."):
        raise HTTPException(status_code=400, detail="Tên file kết quả không hợp lệ")

    base_dir = os.path.abspath(os.path.join(OUTPUT_DIR, image_id))
    path = os.path.abspath(os.path.join(base_dir, safe_path))
    if not path.startswith(base_dir + os.sep):
        raise HTTPException(status_code=400, detail="Tên file kết quả không hợp lệ")
    return path


def _pick_ocr_source(item: Dict[str, Any], source_filename: Optional[str] = None):
    if source_filename:
        path = _safe_output_path(item["id"], source_filename)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Không tìm thấy ảnh kết quả để OCR")
        return path, source_filename

    candidates = [
        result
        for result in item.get("results", [])
        if (result.get("type") in {None, "image"}) and result.get("filename")
    ]
    if not candidates:
        raise HTTPException(status_code=400, detail="Chưa có ảnh kết quả để OCR")

    preferred_steps = ("enhance_adaptive", "yolo_enhance_adaptive", "enhance_otsu", "yolo_enhance_otsu")
    for step_name in preferred_steps:
        for result in reversed(candidates):
            if result.get("step") == step_name:
                path = _safe_output_path(item["id"], result["filename"])
                if os.path.isfile(path):
                    return path, result["filename"]

    for result in reversed(candidates):
        path = _safe_output_path(item["id"], result["filename"])
        if os.path.isfile(path):
            return path, result["filename"]

    raise HTTPException(status_code=404, detail="Không tìm thấy ảnh kết quả để OCR")


_load_existing_yolo_model()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/config")
async def get_default_config():
    return DEFAULT_CONFIG


@app.post("/api/model/upload")
async def upload_yolo_model(file: UploadFile = File(...)):
    filename, path, reused = await _save_or_reuse_yolo_model(file)
    return {
        "filename": filename,
        "model_path": path,
        "has_model": True,
        "reused": reused,
        "message": "Đã dùng lại model YOLO đã upload trước đó" if reused else "Đã upload model YOLO",
    }


@app.get("/api/model/status")
async def get_yolo_model_status():
    if not active_yolo_model["path"] or not os.path.exists(active_yolo_model["path"]):
        _load_existing_yolo_model()
    return {
        "filename": active_yolo_model["filename"],
        "has_model": bool(active_yolo_model["path"] and os.path.exists(active_yolo_model["path"])),
    }


@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    uploaded = []
    for file in files:
        original_name = os.path.basename(file.filename or "image.png")

        # Nếu upload lại ảnh cùng tên, xóa ảnh/kết quả cũ để tránh sinh nhiều card trùng nhau.
        for old_item in store.remove_by_filename(original_name):
            _delete_image_files(old_item)

        filename, path = await save_upload_file(UPLOAD_DIR, file)
        item = store.add_image(filename, path)
        item["original_url"] = f"/uploads/{os.path.basename(path)}"
        uploaded.append({
            "id": item["id"],
            "filename": item["filename"],
            "status": item["status"],
            "original_url": item["original_url"],
        })
    return {"uploaded": uploaded}


@app.post("/api/run")
async def run_pipeline(payload: RunRequest, background_tasks: BackgroundTasks):
    if store.is_processing:
        return {"message": "Pipeline đang chạy", "status": "processing"}

    image_ids = payload.image_ids or [image["id"] for image in store.list_images()]
    image_ids = list(dict.fromkeys(image_ids))

    if not image_ids:
        raise HTTPException(status_code=400, detail="Chưa có ảnh nào trong hàng chờ")

    config = payload.config or DEFAULT_CONFIG
    if config.get("processor") == "yolo":
        if not active_yolo_model["path"] or not os.path.exists(active_yolo_model["path"]):
            _load_existing_yolo_model()
        if not active_yolo_model["path"] or not os.path.exists(active_yolo_model["path"]):
            raise HTTPException(status_code=400, detail="Bạn cần upload file YOLO .pt trước khi chạy chế độ YOLO")
        config.setdefault("yolo", {})["model_path"] = active_yolo_model["path"]

    # Xóa kết quả cũ của các ảnh sắp chạy, tránh frontend đọc lại output cũ.
    for image_id in image_ids:
        safe_remove_path(os.path.join(OUTPUT_DIR, image_id))

    store.reset_for_run(image_ids)
    background_tasks.add_task(process_queue, config)
    return {"message": "Đã bắt đầu xử lý", "status": "started", "total": len(image_ids)}


async def process_queue(config: Dict[str, Any]):
    store.is_processing = True
    try:
        while store.queue:
            image_id = store.queue.pop(0)
            item = store.get(image_id)
            if not item:
                continue
            item["status"] = "processing"
            item["progress"] = 10
            await asyncio.sleep(0.05)
            try:
                results = runner.process_image(image_id, item["path"], config)
                item["results"] = results
                item["status"] = "done"
                item["progress"] = 100
            except Exception as exc:
                item["status"] = "error"
                item["progress"] = 100
                item["error"] = str(exc)
    finally:
        store.is_processing = False


@app.get("/api/status")
async def get_status():
    images = []
    seen_filenames = set()
    for item in reversed(store.list_images()):
        # Chỉ trả về bản mới nhất cho mỗi filename để tránh frontend render lặp.
        if item["filename"] in seen_filenames:
            continue
        seen_filenames.add(item["filename"])
        images.append({
            "id": item["id"],
            "filename": item["filename"],
            "status": item["status"],
            "progress": item["progress"],
            "error": item["error"],
            "original_url": item.get("original_url"),
            "result_count": len(item.get("results", [])),
        })
    images.reverse()
    return {"is_processing": store.is_processing, "images": images}


@app.get("/api/results/{image_id}")
async def get_results(image_id: str):
    item = store.get(image_id)
    if not item:
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh")
    return {
        "id": item["id"],
        "filename": item["filename"],
        "status": item["status"],
        "progress": item["progress"],
        "error": item["error"],
        "original_url": item.get("original_url"),
        "results": item.get("results", []),
        "ocr_results": item.get("ocr_results", []),
    }


@app.post("/api/ocr/{image_id}")
async def run_ocr(image_id: str, payload: OcrRequest):
    if store.is_processing:
        raise HTTPException(status_code=409, detail="Pipeline đang chạy, vui lòng đợi xử lý xong rồi OCR")

    item = store.get(image_id)
    if not item:
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh")
    if item.get("status") != "done":
        raise HTTPException(status_code=400, detail="Chỉ OCR sau khi scan xử lý xong")

    source_path, source_filename = _pick_ocr_source(item, payload.source_filename)
    config = payload.config or DEFAULT_CONFIG
    try:
        ocr_results = await asyncio.to_thread(
            ocr_runner.process_result_image,
            image_id,
            source_path,
            config,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    item["ocr_results"] = ocr_results
    item["ocr_source"] = source_filename
    return {
        "message": "Đã OCR bằng MinerU",
        "source_filename": source_filename,
        "ocr_results": ocr_results,
    }


@app.get("/api/download/{image_id}/{file_path:path}")
async def download_result(image_id: str, file_path: str):
    path = _safe_output_path(image_id, file_path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file kết quả")

    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return FileResponse(path, filename=os.path.basename(path), media_type=media_type)


@app.delete("/api/clear")
async def clear_queue():
    if store.is_processing:
        raise HTTPException(status_code=409, detail="Pipeline đang chạy, vui lòng đợi xử lý xong rồi xóa cache")

    store.clear()
    clear_directory(UPLOAD_DIR)
    clear_directory(OUTPUT_DIR)

    return {
        "message": "Đã xóa cache ảnh và kết quả cũ",
        "cleared_uploads": True,
        "cleared_outputs": True,
        "kept_yolo_model": bool(active_yolo_model["path"] and os.path.exists(active_yolo_model["path"])),
    }


@app.delete("/api/cache")
async def clear_cache_alias():
    return await clear_queue()
