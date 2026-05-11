import os
import shutil
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import UploadFile


class JobStore:
    def __init__(self):
        self.images: Dict[str, Dict] = {}
        self.queue: List[str] = []
        self.is_processing = False

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
            "error": None,
        }
        self.images[image_id] = item
        self.queue.append(image_id)
        return item

    def list_images(self):
        return list(self.images.values())

    def get(self, image_id):
        return self.images.get(image_id)

    def reset_for_run(self, image_ids=None):
        target_ids = image_ids or list(self.images.keys())
        self.queue = []
        seen = set()
        for image_id in target_ids:
            if image_id in seen:
                continue
            seen.add(image_id)
            image = self.images.get(image_id)
            if image:
                image["status"] = "queued"
                image["progress"] = 0
                image["results"] = []
                image["error"] = None
                self.queue.append(image_id)

    def remove(self, image_id: str) -> Optional[Dict]:
        item = self.images.pop(image_id, None)
        self.queue = [queued_id for queued_id in self.queue if queued_id != image_id]
        return item

    def remove_by_filename(self, filename: str) -> List[Dict]:
        removed = []
        for image_id, item in list(self.images.items()):
            if item.get("filename") == filename:
                removed_item = self.remove(image_id)
                if removed_item:
                    removed.append(removed_item)
        return removed

    def clear(self):
        self.images.clear()
        self.queue.clear()
        self.is_processing = False


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
    with open(path, "wb") as f:
        f.write(content)
    return safe_name, path
