import os
import shutil
from typing import Dict

import cv2
import numpy as np


DEFAULT_CONFIG = {
    "processor": "opencv",
    "mode": "color",
    "illumination_method": "lab",
    "gray_equalization_method": "clahe",
    "edge_method": "sobel",
    "morph_operation": "closing",
    "steps": {
        "gaussian_blur": True,
        "median_blur": True,
        "sharpen": True,
        "illumination": True,
        "edge_detection": True,
        "morphology": True,
        "find_contour": True,
        "detect_corners": True,
        "perspective_transform": True,
        "enhance": True,
    },
    "params": {
        "gaussian_ksize": 3,
        "median_ksize": 3,
        "sharpen_amount": 1.0,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": 8,
        "sobel_ksize": 3,
        "sobel_threshold": 30,
        "canny_low": 50,
        "canny_high": 150,
        "morph_kernel": 3,
        "morph_iterations": 1,
        "contour_min_area_ratio": 0.02,
        "enhance_alpha": 1.15,
        "enhance_beta": 8,
        "enhance_method": "otsu",
        "otsu_blur_ksize": 3,
        "adaptive_block_size": 31,
        "adaptive_c": 7,
        "show_both_thresholds": False,
        "yolo_confidence": 0.25,
        "yolo_mask_threshold": 0.5,
        "yolo_gaussian_ksize": 3,
        "yolo_median_ksize": 3,
        "yolo_sharpen_amount": 1.0,
        "yolo_enhance_method": "otsu",
        "yolo_otsu_blur_ksize": 3,
        "yolo_adaptive_block_size": 31,
        "yolo_adaptive_c": 7,
        "yolo_show_both_thresholds": False,
        "yolo_warp_source": "preprocessed",
        "hough_canny_low": 50,
        "hough_canny_high": 150,
        "hough_threshold": 80,
        "hough_min_line_length_ratio": 0.25,
        "hough_max_line_gap": 20,
        "hough_line_thickness": 4,
        "hough_morph_kernel": 9,
        "hough_morph_iterations": 2,
        "hough_gaussian_ksize": 3,
        "hough_median_ksize": 3,
        "hough_sharpen_amount": 1.0,
        "hough_enhance_method": "otsu",
        "hough_otsu_blur_ksize": 3,
        "hough_adaptive_block_size": 31,
        "hough_adaptive_c": 7,
        "hough_show_both_thresholds": False,
        "hough_warp_source": "preprocessed",
        "manual_enhance_method": "otsu",
        "manual_otsu_blur_ksize": 3,
        "manual_adaptive_block_size": 31,
        "manual_adaptive_c": 7,
        "manual_show_both_thresholds": False,
    },
    "yolo_steps": {
        "gaussian_blur": False,
        "median_blur": True,
        "sharpen": False,
        "illumination": False,
        "enhance": True,
    },
    "yolo": {
        "model_path": "",
    },
    "hough_steps": {
        "gaussian_blur": True,
        "median_blur": True,
        "sharpen": False,
        "illumination": True,
        "morphology": True,
        "enhance": True,
    },
    "manual_steps": {
        "enhance": True,
    },
    "manual": {
        "corners_by_image": {},
    },
    "mineru": {
        "method": "ocr",
        "backend": "pipeline",
        "lang": "",
        "formula": True,
        "table": True,
        "timeout_seconds": 3600,
    },
}


class SegmentationBase:
    def __init__(self, config=None):
        self.config = self._merge_config(DEFAULT_CONFIG, config or {})

    def run(self, image):
        images, names = self.step_by_step(image)
        return images[-1] if images else image

    def run_batch(self, images):
        return [self.run(image) for image in images]

    def step_by_step(self, image):
        mode = self.config.get("mode", "color")
        if mode == "gray":
            return self._gray_pipeline(image)
        return self._color_pipeline(image)

    def _merge_config(self, base, update):
        result = {}
        for key, value in base.items():
            if isinstance(value, dict):
                result[key] = self._merge_config(value, update.get(key, {}))
            else:
                result[key] = update.get(key, value)
        for key, value in update.items():
            if key not in result:
                result[key] = value
        return result

    def _enabled(self, step_name):
        steps = self.config.get("steps", {})
        if not any(steps.values()):
            return DEFAULT_CONFIG["steps"].get(step_name, True)
        return bool(steps.get(step_name, False))

    def _param(self, name):
        return self.config.get("params", {}).get(name, DEFAULT_CONFIG["params"].get(name))

    def _odd(self, value, minimum=3):
        value = int(value)
        value = max(value, minimum)
        return value if value % 2 == 1 else value + 1

    def _color_pipeline(self, image):
        current = image.copy()
        steps, names = [current.copy()], ["original"]
        edge_map = None
        contour = None
        corners = None

        if self._enabled("gaussian_blur"):
            k = self._odd(self._param("gaussian_ksize"))
            current = cv2.GaussianBlur(current, (k, k), 0)
            self._add(steps, names, current, "gaussian_blur")

        if self._enabled("median_blur"):
            k = self._odd(self._param("median_ksize"))
            current = cv2.medianBlur(current, k)
            self._add(steps, names, current, "median_blur")

        if self._enabled("sharpen"):
            current = self._sharpen(current)
            self._add(steps, names, current, "sharpen")

        if self._enabled("illumination"):
            method = self.config.get("illumination_method", "lab")
            current = self._equalize_color(current, method)
            self._add(steps, names, current, f"illumination_{method}")

        if self._enabled("edge_detection"):
            method = self.config.get("edge_method", "canny")
            edge_map = self._edge_detection(current, method)
            self._add(steps, names, edge_map, f"edge_{method}")

        if self._enabled("morphology"):
            if edge_map is None:
                edge_map = self._edge_detection(current, self.config.get("edge_method", "canny"))
            edge_map = self._morphology(edge_map)
            self._add(steps, names, edge_map, "morphology")

        if self._enabled("find_contour"):
            if edge_map is None:
                edge_map = self._edge_detection(current, self.config.get("edge_method", "canny"))
            contour = self._find_document_contour(edge_map)
            vis = self._draw_contour(current, contour)
            self._add(steps, names, vis, "find_contour")

        if self._enabled("detect_corners"):
            if contour is None:
                if edge_map is None:
                    edge_map = self._edge_detection(current, self.config.get("edge_method", "canny"))
                contour = self._find_document_contour(edge_map)
            corners = self._detect_corners(contour)
            vis = self._draw_corners(current, corners)
            self._add(steps, names, vis, "detect_corners")

        if self._enabled("perspective_transform"):
            if corners is None:
                if contour is None:
                    if edge_map is None:
                        edge_map = self._edge_detection(current, self.config.get("edge_method", "canny"))
                    contour = self._find_document_contour(edge_map)
                corners = self._detect_corners(contour)
            current = self._perspective_transform(current, corners)
            self._add(steps, names, current, "perspective_transform")

        if self._enabled("enhance"):
            current = self._add_enhance_steps(steps, names, current)

        return steps, names

    def _gray_pipeline(self, image):
        if len(image.shape) == 3:
            current = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            current = image.copy()
        steps, names = [current.copy()], ["grayscale"]
        edge_map = None
        contour = None
        corners = None

        if self._enabled("gaussian_blur"):
            k = self._odd(self._param("gaussian_ksize"))
            current = cv2.GaussianBlur(current, (k, k), 0)
            self._add(steps, names, current, "gaussian_blur")

        if self._enabled("median_blur"):
            k = self._odd(self._param("median_ksize"))
            current = cv2.medianBlur(current, k)
            self._add(steps, names, current, "median_blur")

        if self._enabled("sharpen"):
            current = self._sharpen_gray(current)
            self._add(steps, names, current, "sharpen")

        if self._enabled("illumination"):
            method = self.config.get("gray_equalization_method", "clahe")
            current = self._equalize_gray(current, method)
            self._add(steps, names, current, f"illumination_{method}")

        if self._enabled("edge_detection"):
            method = self.config.get("edge_method", "canny")
            edge_map = self._edge_detection_gray(current, method)
            self._add(steps, names, edge_map, f"edge_{method}")

        if self._enabled("morphology"):
            if edge_map is None:
                edge_map = self._edge_detection_gray(current, self.config.get("edge_method", "canny"))
            edge_map = self._morphology(edge_map)
            self._add(steps, names, edge_map, "morphology")

        source_for_draw = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)

        if self._enabled("find_contour"):
            if edge_map is None:
                edge_map = self._edge_detection_gray(current, self.config.get("edge_method", "canny"))
            contour = self._find_document_contour(edge_map)
            vis = self._draw_contour(source_for_draw, contour)
            self._add(steps, names, vis, "find_contour")

        if self._enabled("detect_corners"):
            if contour is None:
                if edge_map is None:
                    edge_map = self._edge_detection_gray(current, self.config.get("edge_method", "canny"))
                contour = self._find_document_contour(edge_map)
            corners = self._detect_corners(contour)
            vis = self._draw_corners(source_for_draw, corners)
            self._add(steps, names, vis, "detect_corners")

        if self._enabled("perspective_transform"):
            if corners is None:
                if contour is None:
                    if edge_map is None:
                        edge_map = self._edge_detection_gray(current, self.config.get("edge_method", "canny"))
                    contour = self._find_document_contour(edge_map)
                corners = self._detect_corners(contour)
            current = self._perspective_transform(current, corners)
            self._add(steps, names, current, "perspective_transform")

        if self._enabled("enhance"):
            current = self._add_enhance_steps(steps, names, current)

        return steps, names

    def _add(self, steps, names, image, name):
        steps.append(image.copy())
        names.append(name)

    def _sharpen(self, image):
        amount = float(self._param("sharpen_amount"))
        blur = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1 + amount, blur, -amount, 0)

    def _sharpen_gray(self, image):
        amount = float(self._param("sharpen_amount"))
        blur = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1 + amount, blur, -amount, 0)

    def _equalize_color(self, image, method):
        if method == "hsv":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = self._apply_clahe(v)
            return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._apply_clahe(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _equalize_gray(self, image, method):
        if method == "he":
            return cv2.equalizeHist(image)
        return self._apply_clahe(image)

    def _apply_clahe(self, channel):
        clip = float(self._param("clahe_clip_limit"))
        tile = int(self._param("clahe_tile_grid"))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        return clahe.apply(channel)

    def _edge_detection(self, image, method):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return self._edge_detection_gray(gray, method)

    def _edge_detection_gray(self, gray, method):
        if method == "sobel":
            k = self._odd(self._param("sobel_ksize"))
            t = int(self._param("sobel_threshold"))
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, binary = cv2.threshold(mag, t, 255, cv2.THRESH_BINARY)
            return binary
        low = int(self._param("canny_low"))
        high = int(self._param("canny_high"))
        return cv2.Canny(gray, low, high)

    def _morphology(self, image):
        k = int(self._param("morph_kernel"))
        iterations = int(self._param("morph_iterations"))
        op = self.config.get("morph_operation", "closing")
        kernel = np.ones((k, k), np.uint8)
        if op == "erosion":
            return cv2.erode(image, kernel, iterations=iterations)
        if op == "dilation":
            return cv2.dilate(image, kernel, iterations=iterations)
        if op == "opening":
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    def _find_document_contour(self, edge_map):
        contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        h, w = edge_map.shape[:2]
        min_area = h * w * float(self._param("contour_min_area_ratio"))
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                return approx
        return contours[0]

    def _detect_corners(self, contour):
        if contour is None:
            return None
        points = contour.reshape(-1, 2).astype(np.float32)
        if len(points) == 4:
            return self._order_points(points)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect).astype(np.float32)
        return self._order_points(box)

    def _order_points(self, points):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1).reshape(-1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        return rect

    def _perspective_transform(self, image, corners):
        if corners is None:
            return image
        tl, tr, br, bl = corners
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_width = max(int(width_a), int(width_b), 1)
        max_height = max(int(height_a), int(height_b), 1)
        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype=np.float32,
        )
        m = cv2.getPerspectiveTransform(corners, dst)
        return cv2.warpPerspective(image, m, (max_width, max_height))

    def _draw_contour(self, image, contour):
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        if contour is not None:
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3)
        return vis

    def _draw_corners(self, image, corners):
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        if corners is not None:
            pts = corners.astype(int)
            for idx, point in enumerate(pts):
                cv2.circle(vis, tuple(point), 8, (0, 0, 255), -1)
                cv2.putText(vis, str(idx + 1), tuple(point + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.polylines(vis, [pts], True, (255, 0, 0), 2)
        return vis

    def _enhance(self, image, method=None, prefix=""):
        method = method or str(self._param(f"{prefix}enhance_method") or self._param("enhance_method") or "contrast")

        if method in {"otsu", "adaptive"}:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

            if method == "otsu":
                k = self._odd(self._param(f"{prefix}otsu_blur_ksize") or self._param("otsu_blur_ksize"), minimum=1)
                if k > 1:
                    gray = cv2.GaussianBlur(gray, (k, k), 0)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary

            block_size = self._odd(self._param(f"{prefix}adaptive_block_size") or self._param("adaptive_block_size"), minimum=3)
            c_value = int(self._param(f"{prefix}adaptive_c") or self._param("adaptive_c") or 7)
            return cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c_value,
            )

        alpha = float(self._param("enhance_alpha"))
        beta = int(self._param("enhance_beta"))
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        if len(enhanced.shape) == 2:
            return self._apply_clahe(enhanced)
        return enhanced


    def _bool_param(self, name, default=False):
        value = self._param(name)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "có", "co"}
        return bool(value)

    def _add_enhance_steps(self, steps, names, image, prefix="", method_param="enhance_method", show_both_param="show_both_thresholds"):
        show_both = self._bool_param(show_both_param, False)
        method = str(self._param(method_param) or "otsu")

        if show_both:
            otsu = self._enhance(image, method="otsu", prefix=prefix)
            adaptive = self._enhance(image, method="adaptive", prefix=prefix)
            self._add(steps, names, otsu, f"{prefix}enhance_otsu" if prefix else "enhance_otsu")
            self._add(steps, names, adaptive, f"{prefix}enhance_adaptive" if prefix else "enhance_adaptive")
            return adaptive

        enhanced = self._enhance(image, method=method, prefix=prefix)
        step_name = f"{prefix}enhance_{method}" if prefix else f"enhance_{method}"
        self._add(steps, names, enhanced, step_name)
        return enhanced


class YoloDocumentSegmenter(SegmentationBase):
    def _yolo_step_enabled(self, step_name):
        yolo_steps = self.config.get("yolo_steps", {})
        if not yolo_steps:
            return DEFAULT_CONFIG["yolo_steps"].get(step_name, False)
        return bool(yolo_steps.get(step_name, DEFAULT_CONFIG["yolo_steps"].get(step_name, False)))

    def _yolo_param(self, name, fallback_name=None):
        params = self.config.get("params", {})
        if name in params:
            return params.get(name)
        if fallback_name and fallback_name in params:
            return params.get(fallback_name)
        return DEFAULT_CONFIG["params"].get(name, DEFAULT_CONFIG["params"].get(fallback_name))

    def step_by_step_yolo(self, image):
        model_path = self.config.get("yolo", {}).get("model_path") or self.config.get("model_path")
        if not model_path:
            raise ValueError("Chưa upload file model YOLO .pt")
        if not os.path.exists(model_path):
            raise ValueError("Không tìm thấy file model YOLO .pt")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Bạn cần cài ultralytics: pip install ultralytics") from exc

        original = image.copy()
        current = image.copy()
        steps, names = [original.copy()], ["original"]

        # 1) Tiền xử lý ảnh trước khi đưa vào YOLO.
        # Ảnh này cũng có thể được dùng để warp nếu yolo_warp_source = preprocessed.
        if self._yolo_step_enabled("gaussian_blur"):
            k = self._odd(self._yolo_param("yolo_gaussian_ksize", "gaussian_ksize"))
            current = cv2.GaussianBlur(current, (k, k), 0)
            self._add(steps, names, current, "yolo_gaussian_blur")

        if self._yolo_step_enabled("median_blur"):
            k = self._odd(self._yolo_param("yolo_median_ksize", "median_ksize"))
            current = cv2.medianBlur(current, k)
            self._add(steps, names, current, "yolo_median_blur")

        if self._yolo_step_enabled("sharpen"):
            old_amount = self.config.get("params", {}).get("sharpen_amount")
            self.config.setdefault("params", {})["sharpen_amount"] = self._yolo_param("yolo_sharpen_amount", "sharpen_amount")
            current = self._sharpen(current)
            if old_amount is None:
                self.config.get("params", {}).pop("sharpen_amount", None)
            else:
                self.config["params"]["sharpen_amount"] = old_amount
            self._add(steps, names, current, "yolo_sharpen")

        if self._yolo_step_enabled("illumination"):
            method = self.config.get("illumination_method", "lab")
            current = self._equalize_color(current, method)
            self._add(steps, names, current, f"yolo_illumination_{method}")

        conf = float(self._param("yolo_confidence") or 0.25)
        mask_threshold = float(self._param("yolo_mask_threshold") or 0.5)

        # 2) YOLO dự đoán mask trên ảnh đã tiền xử lý.
        prediction = YOLO(model_path).predict(source=current, conf=conf, verbose=False)
        if not prediction:
            raise ValueError("YOLO không trả về kết quả dự đoán")

        result = prediction[0]

        # 3) Từ mask YOLO lấy biên ngoài lớn nhất.
        contour = self._contour_from_yolo_result(result, current.shape[:2], mask_threshold)
        if contour is None:
            raise ValueError("YOLO không tìm thấy mask/box hợp lệ để segment")

        mask_vis = self._mask_from_contour(current.shape[:2], contour)
        self._add(steps, names, mask_vis, "yolo_mask")

        contour_vis = self._draw_contour(current, contour)
        self._add(steps, names, contour_vis, "yolo_outer_contour")

        # 4) Từ biên mask chọn 4 góc rồi chuyển đổi góc nhìn.
        corners = self._detect_corners(contour)
        corners_vis = self._draw_corners(current, corners)
        self._add(steps, names, corners_vis, "yolo_detect_4_corners")

        warp_source = str(self._param("yolo_warp_source") or "preprocessed")
        source_for_warp = current if warp_source == "preprocessed" else original
        warped = self._perspective_transform(source_for_warp, corners)
        self._add(steps, names, warped, "perspective_transform")

        # 5) Tăng cường sau khi đã warp: contrast/otsu/adaptive hoặc trực quan cả Otsu và Adaptive.
        if self._yolo_step_enabled("enhance"):
            warped = self._add_enhance_steps(
                steps,
                names,
                warped,
                prefix="yolo_",
                method_param="yolo_enhance_method",
                show_both_param="yolo_show_both_thresholds",
            )

        return steps, names

    def _contour_from_yolo_result(self, result, image_hw, mask_threshold):
        h, w = image_hw

        # Ưu tiên YOLO segmentation mask: resize mask về kích thước ảnh, threshold,
        # lấy contour ngoài cùng lớn nhất rồi dùng contour đó để tìm 4 góc.
        if getattr(result, "masks", None) is not None and result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy()
            best_contour = None
            best_area = 0

            for mask in masks:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                binary = (mask >= mask_threshold).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > best_area:
                        best_area = area
                        best_contour = contour

            if best_contour is not None and best_area > 0:
                return best_contour

        # Fallback cho trường hợp model chỉ detect box, không có mask.
        if getattr(result, "boxes", None) is not None and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
            return np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)

        return None

    def _mask_from_contour(self, image_hw, contour):
        h, w = image_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, -1)
        return mask


class HoughDocumentSegmenter(SegmentationBase):
    def _hough_step_enabled(self, step_name):
        hough_steps = self.config.get("hough_steps", {})
        if not hough_steps:
            return DEFAULT_CONFIG["hough_steps"].get(step_name, False)
        return bool(hough_steps.get(step_name, DEFAULT_CONFIG["hough_steps"].get(step_name, False)))

    def _hough_param(self, name, fallback_name=None):
        params = self.config.get("params", {})
        if name in params:
            return params.get(name)
        if fallback_name and fallback_name in params:
            return params.get(fallback_name)
        return DEFAULT_CONFIG["params"].get(name, DEFAULT_CONFIG["params"].get(fallback_name))

    def step_by_step_hough(self, image):
        original = image.copy()
        current = image.copy()
        steps, names = [original.copy()], ["original"]

        if self._hough_step_enabled("gaussian_blur"):
            k = self._odd(self._hough_param("hough_gaussian_ksize", "gaussian_ksize"))
            current = cv2.GaussianBlur(current, (k, k), 0)
            self._add(steps, names, current, "hough_gaussian_blur")

        if self._hough_step_enabled("median_blur"):
            k = self._odd(self._hough_param("hough_median_ksize", "median_ksize"))
            current = cv2.medianBlur(current, k)
            self._add(steps, names, current, "hough_median_blur")

        if self._hough_step_enabled("sharpen"):
            old_amount = self.config.get("params", {}).get("sharpen_amount")
            self.config.setdefault("params", {})["sharpen_amount"] = self._hough_param("hough_sharpen_amount", "sharpen_amount")
            current = self._sharpen(current)
            if old_amount is None:
                self.config.get("params", {}).pop("sharpen_amount", None)
            else:
                self.config["params"]["sharpen_amount"] = old_amount
            self._add(steps, names, current, "hough_sharpen")

        if self._hough_step_enabled("illumination"):
            method = self.config.get("illumination_method", "lab")
            current = self._equalize_color(current, method)
            self._add(steps, names, current, f"hough_illumination_{method}")

        gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if len(current.shape) == 3 else current.copy()
        low = int(self._hough_param("hough_canny_low", "canny_low") or 50)
        high = int(self._hough_param("hough_canny_high", "canny_high") or 150)
        edges = cv2.Canny(gray, low, high)
        self._add(steps, names, edges, "hough_canny_edges")

        lines = self._detect_hough_lines(edges)
        line_vis = self._draw_hough_lines(current, lines)
        self._add(steps, names, line_vis, "hough_lines")

        line_mask = self._line_mask(edges.shape[:2], lines)
        self._add(steps, names, line_mask, "hough_line_mask")

        search_mask = line_mask
        if self._hough_step_enabled("morphology"):
            search_mask = self._hough_morphology(search_mask)
            self._add(steps, names, search_mask, "hough_morphology")

        contour = self._find_hough_document_contour(search_mask)
        if contour is None:
            contour = self._find_document_contour(edges)
        if contour is None:
            raise ValueError("Hough Transform không tìm thấy biên tài liệu hợp lệ")

        contour_vis = self._draw_contour(current, contour)
        self._add(steps, names, contour_vis, "hough_outer_contour")

        corners = self._detect_corners(contour)
        corners_vis = self._draw_corners(current, corners)
        self._add(steps, names, corners_vis, "hough_detect_4_corners")

        warp_source = str(self._hough_param("hough_warp_source") or "preprocessed")
        source_for_warp = current if warp_source == "preprocessed" else original
        warped = self._perspective_transform(source_for_warp, corners)
        self._add(steps, names, warped, "perspective_transform")

        if self._hough_step_enabled("enhance"):
            warped = self._add_enhance_steps(
                steps,
                names,
                warped,
                prefix="hough_",
                method_param="hough_enhance_method",
                show_both_param="hough_show_both_thresholds",
            )

        return steps, names

    def _detect_hough_lines(self, edges):
        h, w = edges.shape[:2]
        threshold = int(self._hough_param("hough_threshold") or 80)
        min_line_ratio = float(self._hough_param("hough_min_line_length_ratio") or 0.25)
        min_line_length = max(20, int(min(h, w) * min_line_ratio))
        max_line_gap = int(self._hough_param("hough_max_line_gap") or 20)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        return [] if lines is None else lines.reshape(-1, 4)

    def _draw_hough_lines(self, image, lines):
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in lines:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        return vis

    def _line_mask(self, image_hw, lines):
        h, w = image_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        thickness = int(self._hough_param("hough_line_thickness") or 4)
        for x1, y1, x2, y2 in lines:
            cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness)
        return mask

    def _hough_morphology(self, image):
        k = int(self._hough_param("hough_morph_kernel") or 9)
        iterations = int(self._hough_param("hough_morph_iterations") or 2)
        kernel = np.ones((max(1, k), max(1, k)), np.uint8)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return cv2.dilate(closed, kernel, iterations=1)

    def _find_hough_document_contour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = mask.shape[:2]
        min_area = h * w * float(self._param("contour_min_area_ratio"))
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            return approx if len(approx) == 4 else contour
        return None


class ManualDocumentSegmenter(SegmentationBase):
    def _manual_step_enabled(self, step_name):
        manual_steps = self.config.get("manual_steps", {})
        if not manual_steps:
            return DEFAULT_CONFIG["manual_steps"].get(step_name, False)
        return bool(manual_steps.get(step_name, DEFAULT_CONFIG["manual_steps"].get(step_name, False)))

    def _manual_corners(self, image_id, image_shape):
        corners_by_image = self.config.get("manual", {}).get("corners_by_image", {})
        points = corners_by_image.get(image_id)
        if not points or len(points) != 4:
            raise ValueError("Bạn cần chọn đủ 4 góc thủ công cho ảnh này")

        h, w = image_shape[:2]
        corners = []
        for point in points:
            x = float(point.get("x", 0))
            y = float(point.get("y", 0))
            if 0 <= x <= 1 and 0 <= y <= 1:
                corners.append([x * w, y * h])
            else:
                corners.append([x, y])
        return self._order_points(np.array(corners, dtype=np.float32))

    def step_by_step_manual(self, image_id, image):
        original = image.copy()
        steps, names = [original.copy()], ["original"]

        corners = self._manual_corners(image_id, original.shape)
        contour = corners.reshape(-1, 1, 2).astype(np.float32)
        contour_vis = self._draw_contour(original, contour.astype(np.int32))
        self._add(steps, names, contour_vis, "manual_contour")

        corners_vis = self._draw_corners(original, corners)
        self._add(steps, names, corners_vis, "manual_detect_4_corners")

        warped = self._perspective_transform(original, corners)
        self._add(steps, names, warped, "perspective_transform")

        if self._manual_step_enabled("enhance"):
            warped = self._add_enhance_steps(
                steps,
                names,
                warped,
                prefix="manual_",
                method_param="manual_enhance_method",
                show_both_param="manual_show_both_thresholds",
            )

        return steps, names


class SegmentationRunner:
    def __init__(self, output_root: str):
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

    def process_image(self, image_id: str, image_path: str, config: Dict):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Không thể đọc ảnh đầu vào")
        if (config or {}).get("processor") == "yolo":
            processor = YoloDocumentSegmenter(config)
            images, names = processor.step_by_step_yolo(image)
        elif (config or {}).get("processor") == "hough":
            processor = HoughDocumentSegmenter(config)
            images, names = processor.step_by_step_hough(image)
        elif (config or {}).get("processor") == "manual":
            processor = ManualDocumentSegmenter(config)
            images, names = processor.step_by_step_manual(image_id, image)
        else:
            processor = SegmentationBase(config)
            images, names = processor.step_by_step(image)
        image_output_dir = os.path.join(self.output_root, image_id)
        if os.path.exists(image_output_dir):
            shutil.rmtree(image_output_dir, ignore_errors=True)
        os.makedirs(image_output_dir, exist_ok=True)

        results = []
        seen_steps = set()
        for idx, (step_image, step_name) in enumerate(zip(images, names)):
            filename = f"{idx:02d}_{step_name}.png"
            path = os.path.join(image_output_dir, filename)
            cv2.imwrite(path, step_image)

            # Ảnh gốc đã được frontend hiển thị riêng bằng original_url.
            # Không đưa original/grayscale vào results để tránh lặp trong ảnh trung gian.
            if step_name in {"original", "grayscale"}:
                continue
            if step_name in seen_steps:
                continue
            seen_steps.add(step_name)

            results.append({
                "step": step_name,
                "filename": filename,
                "url": f"/outputs/{image_id}/{filename}",
                "download_url": f"/api/download/{image_id}/{filename}",
            })
        return results
