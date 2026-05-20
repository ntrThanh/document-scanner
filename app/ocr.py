import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict
from urllib.parse import quote


SUPPORTED_MINERU_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".docx",
    ".pptx",
    ".xlsx",
}


class MinerURunner:
    def __init__(self, output_root: str):
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

    def process_result_image(self, image_id: str, file_path: str, config: Dict):
        image_output_dir = os.path.join(self.output_root, image_id)
        ocr_output_dir = os.path.join(image_output_dir, "ocr")
        if os.path.exists(ocr_output_dir):
            shutil.rmtree(ocr_output_dir, ignore_errors=True)
        os.makedirs(ocr_output_dir, exist_ok=True)

        self._run_mineru(file_path, ocr_output_dir, config)
        return self._collect_results(image_id, ocr_output_dir, "ocr")

    def process_file(self, image_id: str, file_path: str, config: Dict):
        image_output_dir = os.path.join(self.output_root, image_id)
        if os.path.exists(image_output_dir):
            shutil.rmtree(image_output_dir, ignore_errors=True)
        os.makedirs(image_output_dir, exist_ok=True)

        self._run_mineru(file_path, image_output_dir, config)
        return self._collect_results(image_id, image_output_dir)

    def _run_mineru(self, file_path: str, output_dir: str, config: Dict):
        suffix = Path(file_path).suffix.lower()
        if suffix not in SUPPORTED_MINERU_SUFFIXES:
            raise ValueError("MinerU chỉ hỗ trợ PDF, ảnh, DOCX, PPTX hoặc XLSX")

        mineru_bin = shutil.which("mineru")
        if not mineru_bin:
            raise RuntimeError(
                'Chưa tìm thấy lệnh "mineru". Cài MinerU bằng: uv pip install -U "mineru[all]"'
            )

        mineru_config = (config or {}).get("mineru", {})
        method = str(mineru_config.get("method") or "ocr")
        backend = str(mineru_config.get("backend") or "pipeline")
        language = str(mineru_config.get("lang") or "").strip()
        timeout = int(mineru_config.get("timeout_seconds") or 3600)
        formula = bool(mineru_config.get("formula", True))
        table = bool(mineru_config.get("table", True))

        cmd = [
            mineru_bin,
            "-p",
            file_path,
            "-o",
            output_dir,
            "-m",
            method,
            "-b",
            backend,
            "-f",
            str(formula).lower(),
            "-t",
            str(table).lower(),
        ]
        if language:
            cmd.extend(["-l", language])

        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"MinerU OCR lỗi: {stderr[-1500:] or 'không có log lỗi'}")

    def _collect_results(self, image_id: str, output_dir: str, url_prefix: str = ""):
        files = []
        for root, _, names in os.walk(output_dir):
            for name in names:
                path = os.path.join(root, name)
                rel_path = os.path.relpath(path, output_dir)
                files.append((self._priority(name), rel_path, path))

        files.sort(key=lambda item: (item[0], item[1].lower()))
        results = []
        for _, rel_path, path in files:
            suffix = Path(path).suffix.lower()
            result_type = self._result_type(suffix)
            item = {
                "step": self._label_for(rel_path, result_type),
                "filename": os.path.join(url_prefix, rel_path) if url_prefix else rel_path,
                "type": result_type,
                "url": f"/outputs/{image_id}/{quote(os.path.join(url_prefix, rel_path))}",
                "download_url": f"/api/download/{image_id}/{quote(os.path.join(url_prefix, rel_path))}",
            }

            if result_type == "markdown":
                item["preview"] = self._read_preview(path)

            results.append(item)

        if not results:
            raise RuntimeError("MinerU đã chạy xong nhưng không tìm thấy file kết quả")
        return results

    def _priority(self, filename: str):
        lower_name = filename.lower()
        if lower_name.endswith(".md"):
            return 0
        if lower_name.endswith("_content_list.json"):
            return 1
        if lower_name.endswith(".json"):
            return 2
        if lower_name.endswith(".pdf"):
            return 3
        if Path(lower_name).suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return 4
        return 5

    def _result_type(self, suffix: str):
        if suffix == ".md":
            return "markdown"
        if suffix == ".json":
            return "json"
        if suffix == ".pdf":
            return "pdf"
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return "image"
        return "file"

    def _label_for(self, rel_path: str, result_type: str):
        name = os.path.basename(rel_path)
        if result_type == "markdown":
            return "Markdown OCR"
        if name.endswith("_content_list.json"):
            return "Content list JSON"
        if name.endswith("_middle.json"):
            return "Middle JSON"
        if name.endswith("_model.json"):
            return "Model JSON"
        if name.endswith("_layout.pdf"):
            return "Layout PDF"
        if name.endswith("_span.pdf"):
            return "Span PDF"
        return name

    def _read_preview(self, path: str, limit: int = 5000):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(limit + 1)
        except OSError:
            return ""
        return content[:limit] + ("\n..." if len(content) > limit else "")
