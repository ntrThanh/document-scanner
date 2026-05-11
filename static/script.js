(() => {
    if (window.__documentSegmentationAppInitialized) return;
    window.__documentSegmentationAppInitialized = true;

const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const uploadBtn = document.getElementById("uploadBtn");
const runBtn = document.getElementById("runBtn");
const clearBtn = document.getElementById("clearBtn");
const defaultBtn = document.getElementById("defaultBtn");
const selectedFiles = document.getElementById("selectedFiles");
const queueList = document.getElementById("queueList");
const queueSummary = document.getElementById("queueSummary");
const results = document.getElementById("results");
const stepsBox = document.getElementById("stepsBox");
const opencvPanel = document.getElementById("opencvPanel");
const yoloPanel = document.getElementById("yoloPanel");
const modelInput = document.getElementById("modelInput");
const chooseModelBtn = document.getElementById("chooseModelBtn");
const modelStatus = document.getElementById("modelStatus");
const opencvTab = document.getElementById("opencvTab");
const yoloTab = document.getElementById("yoloTab");
const imageViewer = document.getElementById("imageViewer");
const viewerImage = document.getElementById("viewerImage");
const viewerCanvas = document.getElementById("viewerCanvas");
const viewerClose = document.getElementById("viewerClose");
const viewerZoomIn = document.getElementById("viewerZoomIn");
const viewerZoomOut = document.getElementById("viewerZoomOut");
const viewerZoomReset = document.getElementById("viewerZoomReset");

const CONFIG_STORAGE_KEY = "documentSegmentation:lastConfig:v5";

let currentFiles = [];
let statusTimer = null;
let renderedResults = new Set();
let isRunning = false;
let isRefreshingStatus = false;
let lastRunToken = 0;

const stepLabels = {
    gaussian_blur: "Gaussian Blur",
    median_blur: "Median Blur",
    sharpen: "Làm sắc ảnh",
    illumination: "Cân bằng sáng",
    edge_detection: "Phát hiện biên",
    morphology: "Biến đổi hình thái",
    find_contour: "Tìm contour",
    detect_corners: "Phát hiện 4 góc",
    perspective_transform: "Perspective Transform",
    enhance: "Tăng cường ảnh"
};

const defaultConfig = {
    processor: "opencv",
    mode: "color",
    illumination_method: "lab",
    gray_equalization_method: "clahe",
    edge_method: "sobel",
    morph_operation: "closing",
    steps: {
        gaussian_blur: true,
        median_blur: true,
        sharpen: true,
        illumination: true,
        edge_detection: true,
        morphology: true,
        find_contour: true,
        detect_corners: true,
        perspective_transform: true,
        enhance: true
    },
    params: {
        gaussian_ksize: 3,
        median_ksize: 3,
        sharpen_amount: 1.0,
        clahe_clip_limit: 2.0,
        clahe_tile_grid: 8,
        sobel_ksize: 3,
        sobel_threshold: 30,
        canny_low: 50,
        canny_high: 150,
        morph_kernel: 3,
        morph_iterations: 1,
        contour_min_area_ratio: 0.02,
        enhance_alpha: 1.15,
        enhance_beta: 8,
        enhance_method: "otsu",
        otsu_blur_ksize: 3,
        adaptive_block_size: 31,
        adaptive_c: 7,
        show_both_thresholds: false,
        yolo_confidence: 0.25,
        yolo_mask_threshold: 0.5,
        yolo_gaussian_ksize: 3,
        yolo_median_ksize: 3,
        yolo_sharpen_amount: 1.0,
        yolo_enhance_method: "otsu",
        yolo_otsu_blur_ksize: 3,
        yolo_adaptive_block_size: 31,
        yolo_adaptive_c: 7,
        yolo_show_both_thresholds: false,
        yolo_warp_source: "preprocessed"
    },
    yolo_steps: {
        gaussian_blur: false,
        median_blur: true,
        sharpen: false,
        illumination: false,
        enhance: true
    },
    yolo: {model_path: ""}
};

function deepMerge(base, update) {
    const result = Array.isArray(base) ? [...base] : {...base};
    Object.entries(update || {}).forEach(([key, value]) => {
        if (value && typeof value === "object" && !Array.isArray(value) && base && typeof base[key] === "object") {
            result[key] = deepMerge(base[key], value);
        } else {
            result[key] = value;
        }
    });
    return result;
}

function getSavedConfig() {
    try {
        const raw = localStorage.getItem(CONFIG_STORAGE_KEY);
        return raw ? deepMerge(defaultConfig, JSON.parse(raw)) : defaultConfig;
    } catch (_) {
        return defaultConfig;
    }
}

function saveConfigToStorage() {
    try {
        localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(buildConfig()));
    } catch (_) {}
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function getElementValue(id, fallback = "") {
    const element = document.getElementById(id);
    return element ? element.value : fallback;
}

function setElementValue(id, value) {
    const element = document.getElementById(id);
    if (element) element.value = value;
}

function getNumberValue(id, fallback = 0) {
    const value = Number(getElementValue(id, fallback));
    return Number.isFinite(value) ? value : fallback;
}

function getProcessor() {
    return document.querySelector('input[name="processor"]:checked')?.value || defaultConfig.processor;
}

function setProcessor(processor) {
    const input = document.querySelector(`input[name="processor"][value="${processor}"]`);
    if (input) input.checked = true;

    const isYolo = processor === "yolo";
    opencvPanel?.classList.toggle("hidden", isYolo);
    yoloPanel?.classList.toggle("hidden", !isYolo);
    opencvTab?.classList.toggle("active", !isYolo);
    yoloTab?.classList.toggle("active", isYolo);
}

function initSteps() {
    if (!stepsBox) return;

    stepsBox.innerHTML = "";
    Object.entries(stepLabels).forEach(([key, label]) => {
        const item = document.createElement("label");
        item.className = "step-item";
        item.innerHTML = `<input type="checkbox" data-step="${key}" checked> <span>${escapeHtml(label)}</span>`;
        stepsBox.appendChild(item);
    });
}

function applyConfigToForm(config = defaultConfig) {
    const params = config.params || defaultConfig.params;

    setProcessor(config.processor || defaultConfig.processor);

    setElementValue("mode", config.mode || defaultConfig.mode);
    setElementValue("illuminationMethod", config.illumination_method || defaultConfig.illumination_method);
    setElementValue("grayEqualizationMethod", config.gray_equalization_method || defaultConfig.gray_equalization_method);
    setElementValue("edgeMethod", config.edge_method || defaultConfig.edge_method);
    setElementValue("morphOperation", config.morph_operation || defaultConfig.morph_operation);

    setElementValue("gaussianKsize", params.gaussian_ksize ?? defaultConfig.params.gaussian_ksize);
    setElementValue("medianKsize", params.median_ksize ?? defaultConfig.params.median_ksize);
    setElementValue("sharpenAmount", params.sharpen_amount ?? defaultConfig.params.sharpen_amount);
    setElementValue("claheClipLimit", params.clahe_clip_limit ?? defaultConfig.params.clahe_clip_limit);
    setElementValue("claheTileGrid", params.clahe_tile_grid ?? defaultConfig.params.clahe_tile_grid);
    setElementValue("sobelKsize", params.sobel_ksize ?? defaultConfig.params.sobel_ksize);
    setElementValue("sobelThreshold", params.sobel_threshold ?? defaultConfig.params.sobel_threshold);
    setElementValue("cannyLow", params.canny_low ?? defaultConfig.params.canny_low);
    setElementValue("cannyHigh", params.canny_high ?? defaultConfig.params.canny_high);
    setElementValue("morphKernel", params.morph_kernel ?? defaultConfig.params.morph_kernel);
    setElementValue("morphIterations", params.morph_iterations ?? defaultConfig.params.morph_iterations);
    setElementValue("contourMinAreaRatio", params.contour_min_area_ratio ?? defaultConfig.params.contour_min_area_ratio);
    setElementValue("enhanceAlpha", params.enhance_alpha ?? defaultConfig.params.enhance_alpha);
    setElementValue("enhanceBeta", params.enhance_beta ?? defaultConfig.params.enhance_beta);
    setElementValue("enhanceMethod", params.enhance_method ?? defaultConfig.params.enhance_method);
    setElementValue("otsuBlurKsize", params.otsu_blur_ksize ?? defaultConfig.params.otsu_blur_ksize);
    setElementValue("adaptiveBlockSize", params.adaptive_block_size ?? defaultConfig.params.adaptive_block_size);
    setElementValue("adaptiveC", params.adaptive_c ?? defaultConfig.params.adaptive_c);
    setElementValue("showBothThresholds", String(params.show_both_thresholds ?? defaultConfig.params.show_both_thresholds));

    setElementValue("yoloConfidence", params.yolo_confidence ?? defaultConfig.params.yolo_confidence);
    setElementValue("yoloMaskThreshold", params.yolo_mask_threshold ?? defaultConfig.params.yolo_mask_threshold);
    setElementValue("yoloGaussianKsize", params.yolo_gaussian_ksize ?? defaultConfig.params.yolo_gaussian_ksize);
    setElementValue("yoloMedianKsize", params.yolo_median_ksize ?? defaultConfig.params.yolo_median_ksize);
    setElementValue("yoloSharpenAmount", params.yolo_sharpen_amount ?? defaultConfig.params.yolo_sharpen_amount);
    setElementValue("yoloIlluminationMethod", config.illumination_method || defaultConfig.illumination_method);
    setElementValue("yoloEnhanceMethod", params.yolo_enhance_method ?? defaultConfig.params.yolo_enhance_method);
    setElementValue("yoloOtsuBlurKsize", params.yolo_otsu_blur_ksize ?? defaultConfig.params.yolo_otsu_blur_ksize);
    setElementValue("yoloAdaptiveBlockSize", params.yolo_adaptive_block_size ?? defaultConfig.params.yolo_adaptive_block_size);
    setElementValue("yoloAdaptiveC", params.yolo_adaptive_c ?? defaultConfig.params.yolo_adaptive_c);
    setElementValue("yoloShowBothThresholds", String(params.yolo_show_both_thresholds ?? defaultConfig.params.yolo_show_both_thresholds));
    setElementValue("yoloWarpSource", params.yolo_warp_source ?? defaultConfig.params.yolo_warp_source);

    document.querySelectorAll("[data-step]").forEach(input => {
        input.checked = Boolean((config.steps || defaultConfig.steps)[input.dataset.step]);
    });

    document.querySelectorAll("[data-yolo-step]").forEach(input => {
        input.checked = Boolean((config.yolo_steps || defaultConfig.yolo_steps)[input.dataset.yoloStep]);
    });
}

function applyDefaultConfig() {
    localStorage.removeItem(CONFIG_STORAGE_KEY);
    applyConfigToForm(defaultConfig);
    saveConfigToStorage();
}

function buildConfig() {
    const processor = getProcessor();
    const steps = {};
    document.querySelectorAll("[data-step]").forEach(input => {
        steps[input.dataset.step] = input.checked;
    });

    const yoloSteps = {};
    document.querySelectorAll("[data-yolo-step]").forEach(input => {
        yoloSteps[input.dataset.yoloStep] = input.checked;
    });

    const params = {
        gaussian_ksize: getNumberValue("gaussianKsize", defaultConfig.params.gaussian_ksize),
        median_ksize: getNumberValue("medianKsize", defaultConfig.params.median_ksize),
        sharpen_amount: getNumberValue("sharpenAmount", defaultConfig.params.sharpen_amount),
        clahe_clip_limit: getNumberValue("claheClipLimit", defaultConfig.params.clahe_clip_limit),
        clahe_tile_grid: getNumberValue("claheTileGrid", defaultConfig.params.clahe_tile_grid),
        sobel_ksize: getNumberValue("sobelKsize", defaultConfig.params.sobel_ksize),
        sobel_threshold: getNumberValue("sobelThreshold", defaultConfig.params.sobel_threshold),
        canny_low: getNumberValue("cannyLow", defaultConfig.params.canny_low),
        canny_high: getNumberValue("cannyHigh", defaultConfig.params.canny_high),
        morph_kernel: getNumberValue("morphKernel", defaultConfig.params.morph_kernel),
        morph_iterations: getNumberValue("morphIterations", defaultConfig.params.morph_iterations),
        contour_min_area_ratio: getNumberValue("contourMinAreaRatio", defaultConfig.params.contour_min_area_ratio),
        enhance_alpha: getNumberValue("enhanceAlpha", defaultConfig.params.enhance_alpha),
        enhance_beta: getNumberValue("enhanceBeta", defaultConfig.params.enhance_beta),
        enhance_method: getElementValue("enhanceMethod", defaultConfig.params.enhance_method),
        otsu_blur_ksize: getNumberValue("otsuBlurKsize", defaultConfig.params.otsu_blur_ksize),
        adaptive_block_size: getNumberValue("adaptiveBlockSize", defaultConfig.params.adaptive_block_size),
        adaptive_c: getNumberValue("adaptiveC", defaultConfig.params.adaptive_c),
        show_both_thresholds: getElementValue("showBothThresholds", String(defaultConfig.params.show_both_thresholds)) === "true",
        yolo_confidence: getNumberValue("yoloConfidence", defaultConfig.params.yolo_confidence),
        yolo_mask_threshold: getNumberValue("yoloMaskThreshold", defaultConfig.params.yolo_mask_threshold),
        yolo_gaussian_ksize: getNumberValue("yoloGaussianKsize", defaultConfig.params.yolo_gaussian_ksize),
        yolo_median_ksize: getNumberValue("yoloMedianKsize", defaultConfig.params.yolo_median_ksize),
        yolo_sharpen_amount: getNumberValue("yoloSharpenAmount", defaultConfig.params.yolo_sharpen_amount),
        yolo_enhance_method: getElementValue("yoloEnhanceMethod", defaultConfig.params.yolo_enhance_method),
        yolo_otsu_blur_ksize: getNumberValue("yoloOtsuBlurKsize", defaultConfig.params.yolo_otsu_blur_ksize),
        yolo_adaptive_block_size: getNumberValue("yoloAdaptiveBlockSize", defaultConfig.params.yolo_adaptive_block_size),
        yolo_adaptive_c: getNumberValue("yoloAdaptiveC", defaultConfig.params.yolo_adaptive_c),
        yolo_show_both_thresholds: getElementValue("yoloShowBothThresholds", String(defaultConfig.params.yolo_show_both_thresholds)) === "true",
        yolo_warp_source: getElementValue("yoloWarpSource", defaultConfig.params.yolo_warp_source)
    };

    return {
        processor,
        mode: getElementValue("mode", defaultConfig.mode),
        illumination_method: processor === "yolo"
            ? getElementValue("yoloIlluminationMethod", defaultConfig.illumination_method)
            : getElementValue("illuminationMethod", defaultConfig.illumination_method),
        gray_equalization_method: getElementValue("grayEqualizationMethod", defaultConfig.gray_equalization_method),
        edge_method: getElementValue("edgeMethod", defaultConfig.edge_method),
        morph_operation: getElementValue("morphOperation", defaultConfig.morph_operation),
        steps,
        yolo_steps: yoloSteps,
        params,
        yolo: {model_path: ""}
    };
}

function updateSelectedFiles(files) {
    currentFiles = Array.from(files || []).filter(file => file.type.startsWith("image/"));
    selectedFiles.innerHTML = currentFiles.length
        ? currentFiles.map(file => `<div>${escapeHtml(file.name)}</div>`).join("")
        : "Chưa chọn ảnh";
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    let data = {};

    try {
        data = await response.json();
    } catch (_) {
        data = {};
    }

    if (!response.ok) {
        throw new Error(data.detail || data.message || `Request lỗi: ${response.status}`);
    }

    return data;
}

function stopPolling() {
    if (statusTimer) clearInterval(statusTimer);
    statusTimer = null;
}

function startPolling() {
    stopPolling();
    statusTimer = setInterval(() => refreshStatus({renderDone: true}), 800);
    refreshStatus({renderDone: true});
}

function setRunningState(running) {
    isRunning = running;
    if (runBtn) {
        runBtn.disabled = running;
        runBtn.textContent = running ? "Đang chạy..." : "Chạy pipeline";
    }
}

function resetResultsView(message = "Chưa có kết quả") {
    renderedResults.clear();
    results.innerHTML = message;
    results.classList.add("empty-state");
}

function getSafeProgress(progress) {
    const value = Number(progress);
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(100, value));
}

function uniqueByUrl(items) {
    const seen = new Set();
    return items.filter(item => {
        const key = item.step || item.filename || item.url || item.download_url;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
}

function withCacheBuster(url) {
    if (!url) return "";
    const separator = url.includes("?") ? "&" : "?";
    return `${url}${separator}t=${Date.now()}`;
}

function cssEscape(value) {
    const text = String(value ?? "");
    if (window.CSS && typeof window.CSS.escape === "function") {
        return window.CSS.escape(text);
    }
    return text.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function removeExistingResultCards(imageId, filename) {
    const selectors = [`[data-result-id="${cssEscape(imageId)}"]`];

    if (filename) {
        selectors.push(`[data-result-filename="${cssEscape(filename)}"]`);
    }

    document.querySelectorAll(selectors.join(",")).forEach(card => card.remove());
}

document.querySelectorAll('input[name="processor"]').forEach(input => {
    input.addEventListener("change", () => setProcessor(input.value));
});

dropZone?.addEventListener("click", () => fileInput?.click());
fileInput?.addEventListener("change", () => updateSelectedFiles(fileInput.files));

dropZone?.addEventListener("dragover", event => {
    event.preventDefault();
    dropZone.classList.add("active");
});

dropZone?.addEventListener("dragleave", () => dropZone.classList.remove("active"));

dropZone?.addEventListener("drop", event => {
    event.preventDefault();
    dropZone.classList.remove("active");
    updateSelectedFiles(event.dataTransfer.files);
});

chooseModelBtn?.addEventListener("click", () => modelInput?.click());

modelInput?.addEventListener("change", async () => {
    const file = modelInput.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pt")) {
        alert("Vui lòng chọn file .pt");
        modelInput.value = "";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        modelStatus.textContent = "Đang upload model...";
        const data = await fetchJson("/api/model/upload", {method: "POST", body: formData});
        modelStatus.textContent = data.reused ? `Đang dùng lại: ${data.filename}` : `Đang dùng: ${data.filename}`;
    } catch (error) {
        modelStatus.textContent = "Upload model thất bại";
        alert(error.message || "Upload model thất bại");
    }
});

uploadBtn?.addEventListener("click", async () => {
    if (!currentFiles.length) {
        alert("Bạn chưa chọn ảnh");
        return;
    }

    const formData = new FormData();
    currentFiles.forEach(file => formData.append("files", file));

    try {
        await fetchJson("/api/upload", {method: "POST", body: formData});
        currentFiles = [];
        fileInput.value = "";
        selectedFiles.innerHTML = "Upload thành công";
        await refreshStatus({renderDone: false});
    } catch (error) {
        alert(error.message || "Upload ảnh thất bại");
    }
});

runBtn?.addEventListener("click", async () => {
    if (isRunning) return;

    try {
        stopPolling();
        setRunningState(true);
        renderedResults.clear();
        lastRunToken += 1;
        results.classList.remove("empty-state");
        results.innerHTML = "";

        await fetchJson("/api/run", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({config: buildConfig()})
        });

        startPolling();
    } catch (error) {
        setRunningState(false);
        alert(error.message || "Không thể chạy pipeline");
    }
});

clearBtn?.addEventListener("click", async () => {
    try {
        stopPolling();
        const data = await fetchJson("/api/clear", {method: "DELETE"});
        queueList.innerHTML = "";
        queueSummary.textContent = "Chưa có ảnh";
        resetResultsView();
        selectedFiles.innerHTML = data.kept_yolo_model ? "Đã xóa cache ảnh/kết quả. Model YOLO vẫn được giữ." : "Đã xóa cache ảnh/kết quả.";
        setRunningState(false);
    } catch (error) {
        alert(error.message || "Không thể xóa dữ liệu");
    }
});

defaultBtn?.addEventListener("click", applyDefaultConfig);

document.addEventListener("change", event => {
    const target = event.target;
    if (!target || !target.matches("input, select")) return;
    if (["fileInput", "modelInput"].includes(target.id)) return;
    saveConfigToStorage();
});

document.addEventListener("input", event => {
    const target = event.target;
    if (!target || !target.matches("input[type='number']")) return;
    saveConfigToStorage();
});

async function refreshStatus(options = {}) {
    if (isRefreshingStatus) return;

    const {renderDone = true} = options;
    isRefreshingStatus = true;

    try {
        const data = await fetchJson("/api/status");
        const images = Array.isArray(data.images) ? data.images : [];

        renderQueue(images);

        if (renderDone) {
            const finishedImages = images.filter(image => image.status === "done" || image.status === "error");

            for (const image of finishedImages) {
                const resultKey = image.filename || image.id;
                if (renderedResults.has(resultKey)) continue;

                renderedResults.add(resultKey);

                if (image.status === "done") {
                    await renderResult(image.id, image.filename);
                }
            }
        }

        if (!data.is_processing && statusTimer) {
            stopPolling();
            setRunningState(false);
        }
    } catch (error) {
        stopPolling();
        setRunningState(false);
        console.error(error);
    } finally {
        isRefreshingStatus = false;
    }
}

function renderQueue(images) {
    queueSummary.textContent = images.length ? `${images.length} ảnh trong danh sách` : "Chưa có ảnh";

    if (!images.length) {
        queueList.innerHTML = "<div class='muted'>Chưa có ảnh nào trong hàng chờ</div>";
        return;
    }

    queueList.innerHTML = images.map(image => {
        const progress = getSafeProgress(image.progress);
        return `
            <div class="queue-item">
                <div class="queue-main">
                    <strong>${escapeHtml(image.filename)}</strong>
                    ${image.error ? `<div class="error-text">${escapeHtml(image.error)}</div>` : ""}
                </div>
                <span class="badge ${escapeHtml(image.status)}">${escapeHtml(image.status)}</span>
                <div class="progress"><span style="width: ${progress}%"></span></div>
            </div>
        `;
    }).join("");
}
async function renderResult(imageId, statusFilename = "") {
    const data = await fetchJson(`/api/results/${encodeURIComponent(imageId)}?t=${Date.now()}`);
    let resultItems = Array.isArray(data.results) ? data.results : [];

    if (!resultItems.length) return;

    const filename = data.filename || statusFilename || `Ảnh ${imageId}`;

    // Chặn lặp original trong ảnh trung gian
    resultItems = resultItems.filter(item => item.step !== "original");

    // Chặn lặp ảnh theo step
    const seenSteps = new Set();
    resultItems = resultItems.filter(item => {
        const key = item.step || item.filename || item.url;
        if (seenSteps.has(key)) return false;
        seenSteps.add(key);
        return true;
    });

    if (!resultItems.length) return;

    results.classList.remove("empty-state");

    // Xóa tất cả card cũ cùng imageId hoặc cùng filename
    document
        .querySelectorAll(
            `[data-result-id="${cssEscape(imageId)}"], [data-result-filename="${cssEscape(filename)}"]`
        )
        .forEach(card => card.remove());

    const card = document.createElement("div");
    card.className = "result-card";
    card.dataset.resultId = String(imageId);
    card.dataset.resultFilename = filename;

    const finalImage = resultItems[resultItems.length - 1];
    const intermediateResults = resultItems.slice(0, -1);

    const intermediateHtml = intermediateResults.map(item => `
        <div class="image-box">
            <img class="zoomable-img" src="${withCacheBuster(item.url)}" data-full-src="${withCacheBuster(item.url)}" alt="${escapeHtml(item.step)}" title="Bấm để xem lớn">
            <h4>${escapeHtml(item.step)}</h4>
            ${item.download_url ? `<a href="${item.download_url}">Tải xuống</a>` : ""}
        </div>
    `).join("");

    const stepsHtml = intermediateResults.length
        ? `
            <details class="accordion result-steps" open>
                <summary>Ảnh trung gian</summary>
                <div class="image-grid">${intermediateHtml}</div>
            </details>
        `
        : "";

    card.innerHTML = `
        <div class="result-header">
            <div>
                <h3>${escapeHtml(filename)}</h3>
                <span class="badge done">done</span>
            </div>
        </div>

        <div class="featured-grid">
            <div class="image-box featured">
                <img class="zoomable-img" src="${withCacheBuster(data.original_url)}" data-full-src="${withCacheBuster(data.original_url)}" alt="original" title="Bấm để xem lớn">
                <h4>Ảnh gốc</h4>
            </div>

            <div class="image-box featured">
                <img class="zoomable-img" src="${withCacheBuster(finalImage.url)}" data-full-src="${withCacheBuster(finalImage.url)}" alt="final" title="Bấm để xem lớn">
                <h4>Kết quả cuối cùng</h4>
                ${finalImage.download_url ? `<a href="${finalImage.download_url}">Tải xuống</a>` : ""}
            </div>
        </div>

        ${stepsHtml}
    `;

    results.prepend(card);
}



let viewerScale = 1;
let viewerX = 0;
let viewerY = 0;
let viewerDragging = false;
let viewerStartX = 0;
let viewerStartY = 0;

function updateViewerTransform() {
    if (!viewerImage) return;
    viewerImage.style.transform = `translate(${viewerX}px, ${viewerY}px) scale(${viewerScale})`;
    if (viewerZoomReset) viewerZoomReset.textContent = `${Math.round(viewerScale * 100)}%`;
}

function openImageViewer(src) {
    if (!imageViewer || !viewerImage || !src) return;
    viewerScale = 1;
    viewerX = 0;
    viewerY = 0;
    viewerImage.src = src;
    imageViewer.classList.remove("hidden");
    imageViewer.setAttribute("aria-hidden", "false");
    updateViewerTransform();
}

function closeImageViewer() {
    if (!imageViewer || !viewerImage) return;
    imageViewer.classList.add("hidden");
    imageViewer.setAttribute("aria-hidden", "true");
    viewerImage.src = "";
}

function zoomViewer(delta) {
    viewerScale = Math.max(0.2, Math.min(8, viewerScale + delta));
    updateViewerTransform();
}

results?.addEventListener("click", event => {
    const img = event.target.closest(".zoomable-img");
    if (!img) return;
    openImageViewer(img.dataset.fullSrc || img.src);
});

viewerClose?.addEventListener("click", closeImageViewer);
viewerZoomIn?.addEventListener("click", () => zoomViewer(0.25));
viewerZoomOut?.addEventListener("click", () => zoomViewer(-0.25));
viewerZoomReset?.addEventListener("click", () => {
    viewerScale = 1;
    viewerX = 0;
    viewerY = 0;
    updateViewerTransform();
});

imageViewer?.addEventListener("click", event => {
    if (event.target === imageViewer) closeImageViewer();
});

viewerCanvas?.addEventListener("wheel", event => {
    event.preventDefault();
    zoomViewer(event.deltaY < 0 ? 0.15 : -0.15);
}, {passive: false});

viewerCanvas?.addEventListener("mousedown", event => {
    viewerDragging = true;
    viewerStartX = event.clientX - viewerX;
    viewerStartY = event.clientY - viewerY;
});

document.addEventListener("mousemove", event => {
    if (!viewerDragging) return;
    viewerX = event.clientX - viewerStartX;
    viewerY = event.clientY - viewerStartY;
    updateViewerTransform();
});

document.addEventListener("mouseup", () => {
    viewerDragging = false;
});

document.addEventListener("keydown", event => {
    if (event.key === "Escape") closeImageViewer();
});

async function loadModelStatus() {
    try {
        const data = await fetchJson("/api/model/status");
        modelStatus.textContent = data.has_model ? `Đang dùng: ${data.filename} (không cần upload lại)` : "Chưa có model";
    } catch (error) {
        modelStatus.textContent = "Không đọc được trạng thái model";
        console.error(error);
    }
}

initSteps();
applyConfigToForm(getSavedConfig());
loadModelStatus();
refreshStatus({renderDone: false});

})();