import os
import shutil
import threading
import uuid
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import UploadFile
from PIL import Image, ImageOps


class JobStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.images: Dict[str, Dict] = {}
        self.queue: List[str] = []
        self.active_jobs = set()

    @property
    def is_processing(self):
        with self.lock:
            return bool(self.active_jobs or self.queue)

    def add_image(self, filename: str, path: str):
        image_id = uuid.uuid4().hex
        item = {
            "id": image_id,
            "filename": filename,
            "path": path,
            "status": "queued",
            "progress": 0,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "results": [],
            "ocr_results": [],
            "ocr_source": None,
            "error": None,
        }
        with self.lock:
            self.images[image_id] = item
            self.queue.append(image_id)
        return item

    def list_images(self):
        with self.lock:
            return [item.copy() for item in self.images.values()]

    def get(self, image_id):
        with self.lock:
            return self.images.get(image_id)

    def update(self, image_id: str, **fields):
        with self.lock:
            item = self.images.get(image_id)
            if item:
                item.update(fields)
            return item

    def has_active(self, image_ids):
        with self.lock:
            return any(image_id in self.active_jobs for image_id in image_ids)

    def reset_for_run(self, image_ids=None):
        with self.lock:
            target_ids = image_ids or list(self.images.keys())
            self.queue = [queued_id for queued_id in self.queue if queued_id not in target_ids]
            queued = []
            seen = set()
            for image_id in target_ids:
                if image_id in seen or image_id in self.active_jobs:
                    continue
                seen.add(image_id)
                image = self.images.get(image_id)
                if image:
                    image["status"] = "queued"
                    image["progress"] = 0
                    image["results"] = []
                    image["ocr_results"] = []
                    image["ocr_source"] = None
                    image["error"] = None
                    self.queue.append(image_id)
                    queued.append(image_id)
            return queued

    def take_next(self):
        with self.lock:
            while self.queue:
                image_id = self.queue.pop(0)
                item = self.images.get(image_id)
                if not item or image_id in self.active_jobs:
                    continue
                self.active_jobs.add(image_id)
                item["status"] = "processing"
                item["progress"] = 10
                item["error"] = None
                return image_id, item.copy()
            return None, None

    def complete(self, image_id: str, results=None, error: Optional[str] = None):
        with self.lock:
            item = self.images.get(image_id)
            if item:
                if error:
                    item["status"] = "error"
                    item["error"] = error
                else:
                    item["status"] = "done"
                    item["results"] = results or []
                    item["error"] = None
                item["progress"] = 100
            self.active_jobs.discard(image_id)

    def remove(self, image_id: str) -> Optional[Dict]:
        with self.lock:
            item = self.images.pop(image_id, None)
            self.queue = [queued_id for queued_id in self.queue if queued_id != image_id]
            self.active_jobs.discard(image_id)
            return item

    def remove_by_filename(self, filename: str) -> List[Dict]:
        removed = []
        with self.lock:
            items = list(self.images.items())
        for image_id, item in items:
            if item.get("filename") == filename and image_id not in self.active_jobs:
                removed_item = self.remove(image_id)
                if removed_item:
                    removed.append(removed_item)
        return removed

    def clear(self):
        with self.lock:
            self.images.clear()
            self.queue.clear()
            self.active_jobs.clear()


def safe_remove_path(path: str):
    if not path:
        return
    if os.path.isfile(path) or os.path.islink(path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    elif os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def clear_directory(directory: str):
    os.makedirs(directory, exist_ok=True)
    for name in os.listdir(directory):
        safe_remove_path(os.path.join(directory, name))


async def save_upload_file(upload_dir: str, file: UploadFile):
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = os.path.basename(file.filename or "image.png")
    ext = os.path.splitext(safe_name)[1].lower() or ".png"
    disk_name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(upload_dir, disk_name)
    content = await file.read()

    try:
        with Image.open(BytesIO(content)) as img:
            normalized = ImageOps.exif_transpose(img)
            save_kwargs = {}
            if ext in {".jpg", ".jpeg"}:
                if normalized.mode not in {"RGB", "L"}:
                    normalized = normalized.convert("RGB")
                save_kwargs = {"quality": 95}
            normalized.save(path, **save_kwargs)
            return safe_name, path
    except Exception:
        pass

    with open(path, "wb") as f:
        f.write(content)
    return safe_name, path
