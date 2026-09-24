/* ═══════════════════════════════════════════════════════════
   Helixo V2 — Client-Side YOLOv8 Inference using ONNX Runtime Web
   + Live Video Feed + Arduino Serial Control
   
   Model:  YOLOv8-nano (best.onnx)
   Input:  [1, 3, 224, 224] (BCHW, float32, 0-1 normalized)
   Output: [1, 6, 1029]  →  6 = [x, y, w, h, helmet_conf, no_helmet_conf]
   ═══════════════════════════════════════════════════════════ */

const MODEL_URL = "./model/best.onnx";
const INPUT_SIZE = 224;
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;
const CLASS_NAMES = ["helmet", "no_helmet"];
const CLASS_COLORS = { helmet: "#10b981", no_helmet: "#ef4444" };

let session = null;

// ─── DOM Elements ─────────────────────────────────────────

// Image Upload Mode
const fileInput = document.getElementById("fileInput");
const uploadZone = document.getElementById("uploadZone");
const fileCountBadge = document.getElementById("fileCountBadge");
const btnDetect = document.getElementById("btnDetect");
const btnWebcam = document.getElementById("btnWebcam");
const processingBar = document.getElementById("processingBar");
const processingText = document.getElementById("processingText");
const placeholder = document.getElementById("placeholder");
const resultsContainer = document.getElementById("resultsContainer");
const webcamBox = document.getElementById("webcamBox");
const webcamVideo = document.getElementById("webcamVideo");
const btnCapture = document.getElementById("btnCapture");
const btnCloseWc = document.getElementById("btnCloseWc");
const procCanvas = document.getElementById("processingCanvas");

// Mode Toggle
const btnModeImage = document.getElementById("btnModeImage");
const btnModeLive = document.getElementById("btnModeLive");
const imageModeContainer = document.getElementById("imageModeContainer");
const liveModeContainer = document.getElementById("liveModeContainer");

// Live Video Mode
const liveVideo = document.getElementById("liveVideo");
const liveCanvas = document.getElementById("liveCanvas");
const liveVideoWrapper = document.getElementById("liveVideoWrapper");
const liveOverlay = document.getElementById("liveOverlay");
const liveLabel = document.getElementById("liveLabel");
const liveConfidence = document.getElementById("liveConfidence");
const btnStartLive = document.getElementById("btnStartLive");
const btnStopLive = document.getElementById("btnStopLive");
const ignitionBadge = document.getElementById("ignitionBadge");
const ignitionText = document.getElementById("ignitionText");
const liveFps = document.getElementById("liveFps");
const liveHelmetCount = document.getElementById("liveHelmetCount");
const liveNoHelmetCount = document.getElementById("liveNoHelmetCount");
const liveTotalFrames = document.getElementById("liveTotalFrames");

// Arduino
const btnArduino = document.getElementById("btnArduino");
const arduinoBtnText = document.getElementById("arduinoBtnText");
const arduinoDot = document.getElementById("arduinoDot");
const arduinoStatusText = document.getElementById("arduinoStatusText");
const arduinoStatusBar = document.getElementById("arduinoStatusBar");

// ─── State ────────────────────────────────────────────────
let currentFiles = [];
let webcamStream = null;

// Live mode
let liveStream = null;
let isLiveRunning = false;
let animFrameId = null;
let liveStats = { helmets: 0, noHelmets: 0, total: 0 };
let fpsCounter = { frames: 0, lastTime: performance.now() };

// Arduino Serial
let serialPort = null;
let serialWriter = null;
let isArduinoConnected = false;

// ═══════════════════════════════════════════════════════════
//  1. LOAD ONNX MODEL
// ═══════════════════════════════════════════════════════════
async function loadModel() {
    const loadingScreen = document.getElementById("modelLoadingScreen");
    const loaderStatus = document.getElementById("loaderStatus");

    try {
        loaderStatus.textContent = "Initializing ONNX Runtime...";
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";

        loaderStatus.textContent = "Downloading YOLOv8-nano model (12 MB)...";
        session = await ort.InferenceSession.create(MODEL_URL, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all",
        });

        loaderStatus.textContent = "Model loaded! ✅";
        setTimeout(() => {
            loadingScreen.classList.add("hidden");
            setTimeout(() => loadingScreen.remove(), 500);
        }, 600);

    } catch (err) {
        loaderStatus.textContent = "❌ Error: " + err.message;
        console.error("Model load error:", err);
    }
}

// ═══════════════════════════════════════════════════════════
//  2. PREPROCESSING & POSTPROCESSING (shared)
// ═══════════════════════════════════════════════════════════
function preprocessImage(imgSource) {
    const ctx = procCanvas.getContext("2d");
    procCanvas.width = INPUT_SIZE;
    procCanvas.height = INPUT_SIZE;

    // Get source dimensions
    const imgW = imgSource.naturalWidth || imgSource.videoWidth || imgSource.width;
    const imgH = imgSource.naturalHeight || imgSource.videoHeight || imgSource.height;
    const scale = Math.min(INPUT_SIZE / imgW, INPUT_SIZE / imgH);
    const newW = Math.round(imgW * scale);
    const newH = Math.round(imgH * scale);
    const padX = (INPUT_SIZE - newW) / 2;
    const padY = (INPUT_SIZE - newH) / 2;

    // Gray background (114/255 is YOLO's default padding)
    ctx.fillStyle = `rgb(114, 114, 114)`;
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    ctx.drawImage(imgSource, padX, padY, newW, newH);

    // Convert to BCHW float32 tensor, normalized 0-1
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const pixels = imageData.data;
    const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        float32Data[i] = pixels[i * 4] / 255.0;
        float32Data[INPUT_SIZE * INPUT_SIZE + i] = pixels[i * 4 + 1] / 255.0;
        float32Data[2 * INPUT_SIZE * INPUT_SIZE + i] = pixels[i * 4 + 2] / 255.0;
    }

    const tensor = new ort.Tensor("float32", float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    return { tensor, scale, padX, padY, imgW, imgH };
}

function postprocess(outputData, scale, padX, padY, imgW, imgH) {
    const numClasses = CLASS_NAMES.length;
    const numBoxes = outputData.length / (4 + numClasses);
    const detections = [];

    for (let i = 0; i < numBoxes; i++) {
        const cx = outputData[0 * numBoxes + i];
        const cy = outputData[1 * numBoxes + i];
        const w  = outputData[2 * numBoxes + i];
        const h  = outputData[3 * numBoxes + i];

        let maxConf = 0;
        let classId = 0;
        for (let c = 0; c < numClasses; c++) {
            const conf = outputData[(4 + c) * numBoxes + i];
            if (conf > maxConf) {
                maxConf = conf;
                classId = c;
            }
        }

        if (maxConf < CONF_THRESHOLD) continue;

        const x1 = ((cx - w / 2) - padX) / scale;
        const y1 = ((cy - h / 2) - padY) / scale;
        const x2 = ((cx + w / 2) - padX) / scale;
        const y2 = ((cy + h / 2) - padY) / scale;

        detections.push({
            x1: Math.max(0, x1), y1: Math.max(0, y1),
            x2: Math.min(imgW, x2), y2: Math.min(imgH, y2),
            confidence: maxConf, classId, className: CLASS_NAMES[classId],
        });
    }

    return nms(detections, IOU_THRESHOLD);
}

function nms(boxes, iouThreshold) {
    boxes.sort((a, b) => b.confidence - a.confidence);
    const kept = [];
    const suppressed = new Set();
    for (let i = 0; i < boxes.length; i++) {
        if (suppressed.has(i)) continue;
        kept.push(boxes[i]);
        for (let j = i + 1; j < boxes.length; j++) {
            if (suppressed.has(j)) continue;
            if (iou(boxes[i], boxes[j]) > iouThreshold) suppressed.add(j);
        }
    }
    return kept;
}

function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-6);
}

async function runInference(imgSource) {
    const { tensor, scale, padX, padY, imgW, imgH } = preprocessImage(imgSource);
    const inputName = session.inputNames[0];
    const results = await session.run({ [inputName]: tensor });
    const outputName = session.outputNames[0];
    return postprocess(results[outputName].data, scale, padX, padY, imgW, imgH);
}

// ═══════════════════════════════════════════════════════════
//  3. LIVE VIDEO MODE
// ═══════════════════════════════════════════════════════════
async function startLiveVideo() {
    if (!session) {
        alert("Please wait until the model is loaded!");
        return;
    }

    try {
        liveStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false,
        });

        liveVideo.srcObject = liveStream;
        await liveVideo.play();

        // Set canvas size to match video
        liveCanvas.width = liveVideo.videoWidth;
        liveCanvas.height = liveVideo.videoHeight;

        isLiveRunning = true;
        liveStats = { helmets: 0, noHelmets: 0, total: 0 };
        fpsCounter = { frames: 0, lastTime: performance.now() };
        updateLiveStatsUI();

        btnStartLive.style.display = "none";
        btnStopLive.style.display = "flex";

        runLiveInferenceLoop();
        console.log("Live video started!");

    } catch (err) {
        console.error("Camera error:", err);
        alert("Could not access camera.\n\nAllow camera permission and try again.\n\nError: " + err.message);
    }
}

function stopLiveVideo() {
    isLiveRunning = false;

    if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }

    if (liveStream) {
        liveStream.getTracks().forEach(t => t.stop());
        liveStream = null;
    }

    liveVideo.srcObject = null;

    // Clear canvas
    const ctx = liveCanvas.getContext("2d");
    ctx.clearRect(0, 0, liveCanvas.width, liveCanvas.height);

    btnStopLive.style.display = "none";
    btnStartLive.style.display = "flex";

    liveLabel.textContent = "Camera Stopped";
    liveLabel.className = "live-label";
    liveConfidence.textContent = "";
    liveVideoWrapper.className = "live-video-wrapper";
    ignitionBadge.className = "ignition-badge";
    ignitionText.textContent = "IGNITION";

    // Safety: cut ignition when stopped
    sendToArduino("N");
    console.log("Live video stopped.");
}

async function runLiveInferenceLoop() {
    if (!isLiveRunning || !session) return;

    try {
        // Run YOLOv8 inference on the live video frame
        const detections = await runInference(liveVideo);

        // Draw bounding boxes on the overlay canvas
        drawLiveBoundingBoxes(detections);

        // Determine overall status
        const helmetCount = detections.filter(d => d.className === "helmet").length;
        const noHelmetCount = detections.filter(d => d.className === "no_helmet").length;

        // Update UI and Arduino
        if (detections.length === 0) {
            updateLiveUI("none", 0);
            sendToArduino("N");
        } else if (helmetCount > 0 && noHelmetCount === 0) {
            const topConf = Math.max(...detections.filter(d => d.className === "helmet").map(d => d.confidence));
            updateLiveUI("helmet", topConf);
            sendToArduino("H");
            liveStats.helmets++;
        } else {
            const topConf = noHelmetCount > 0 
                ? Math.max(...detections.filter(d => d.className === "no_helmet").map(d => d.confidence))
                : 0;
            updateLiveUI("no_helmet", topConf);
            sendToArduino("N");
            liveStats.noHelmets++;
        }

        liveStats.total++;

        // FPS
        fpsCounter.frames++;
        const now = performance.now();
        if (now - fpsCounter.lastTime >= 1000) {
            liveFps.textContent = Math.round(fpsCounter.frames * 1000 / (now - fpsCounter.lastTime));
            fpsCounter.frames = 0;
            fpsCounter.lastTime = now;
        }

        updateLiveStatsUI();

    } catch (err) {
        console.error("Live inference error:", err);
    }

    if (isLiveRunning) {
        animFrameId = requestAnimationFrame(runLiveInferenceLoop);
    }
}

function drawLiveBoundingBoxes(detections) {
    const ctx = liveCanvas.getContext("2d");
    const vw = liveVideo.videoWidth;
    const vh = liveVideo.videoHeight;

    // Match canvas to video size
    if (liveCanvas.width !== vw || liveCanvas.height !== vh) {
        liveCanvas.width = vw;
        liveCanvas.height = vh;
    }

    ctx.clearRect(0, 0, vw, vh);

    detections.forEach(det => {
        const color = CLASS_COLORS[det.className] || "#ffffff";
        const bx = det.x1;
        const by = det.y1;
        const bw = det.x2 - det.x1;
        const bh = det.y2 - det.y1;

        // Draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(3, Math.round(vw / 200));
        ctx.strokeRect(bx, by, bw, bh);

        // Label
        const label = `${det.className.replace("_", " ")} ${Math.round(det.confidence * 100)}%`;
        const fontSize = Math.max(14, Math.round(vw / 35));
        ctx.font = `bold ${fontSize}px Inter, sans-serif`;
        const textW = ctx.measureText(label).width;
        const labelH = fontSize + 10;

        ctx.fillStyle = color;
        ctx.fillRect(bx, by - labelH, textW + 14, labelH);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, bx + 7, by - 6);
    });
}

function updateLiveUI(status, confidence) {
    const ignitionIcon = document.getElementById("ignitionIcon");
    if (status === "helmet") {
        liveLabel.textContent = "HELMET DETECTED ✅";
        liveLabel.className = "live-label helmet";
        liveConfidence.textContent = Math.round(confidence * 100) + "%";
        liveConfidence.style.color = "var(--neon-green)";
        liveVideoWrapper.className = "live-video-wrapper glow-green";
        ignitionBadge.className = "ignition-badge on";
        ignitionText.textContent = "IGNITION ENABLED";
        if (ignitionIcon) ignitionIcon.className = "fas fa-bolt";
    } else if (status === "no_helmet") {
        liveLabel.textContent = "NO HELMET ❌";
        liveLabel.className = "live-label no-helmet";
        liveConfidence.textContent = Math.round(confidence * 100) + "%";
        liveConfidence.style.color = "var(--neon-red)";
        liveVideoWrapper.className = "live-video-wrapper glow-red";
        ignitionBadge.className = "ignition-badge off";
        ignitionText.textContent = "IGNITION CUT";
        if (ignitionIcon) ignitionIcon.className = "fas fa-lock";
    } else {
        liveLabel.textContent = "No Person Detected ⚠️";
        liveLabel.className = "live-label";
        liveConfidence.textContent = "";
        liveVideoWrapper.className = "live-video-wrapper glow-yellow";
        ignitionBadge.className = "ignition-badge off";
        ignitionText.textContent = "IGNITION CUT";
        if (ignitionIcon) ignitionIcon.className = "fas fa-lock";
    }
}

function updateLiveStatsUI() {
    liveHelmetCount.textContent = liveStats.helmets;
    liveNoHelmetCount.textContent = liveStats.noHelmets;
    liveTotalFrames.textContent = liveStats.total;
}

// ═══════════════════════════════════════════════════════════
//  4. WEB SERIAL API — ARDUINO CONNECTION
// ═══════════════════════════════════════════════════════════
async function connectArduino() {
    if (!("serial" in navigator)) {
        alert(
            "⚠️ Web Serial API is NOT supported in this browser!\n\n" +
            "Please use Google Chrome or Microsoft Edge.\n" +
            "Also make sure you're on HTTPS (Netlify) or localhost."
        );
        return;
    }

    if (isArduinoConnected && serialPort) {
        await disconnectArduino();
        return;
    }

    try {
        serialPort = await navigator.serial.requestPort();
        await serialPort.open({ baudRate: 9600 });
        serialWriter = serialPort.writable.getWriter();

        isArduinoConnected = true;
        updateArduinoUI(true);
        console.log("Arduino connected!");

        readArduinoResponses();

    } catch (err) {
        console.error("Arduino connection error:", err);
        if (err.name !== "NotFoundError") {
            alert(
                "Failed to connect to Arduino.\n\n" +
                "Make sure:\n1. Arduino is plugged in via USB\n" +
                "2. Arduino IDE Serial Monitor is CLOSED\n" +
                "3. You selected the correct COM port\n\n" +
                "Error: " + err.message
            );
        }
    }
}

async function disconnectArduino() {
    try {
        if (serialWriter) { serialWriter.releaseLock(); serialWriter = null; }
        if (serialPort) { await serialPort.close(); serialPort = null; }
    } catch (err) { console.error("Disconnect error:", err); }

    isArduinoConnected = false;
    updateArduinoUI(false);
    console.log("Arduino disconnected.");
}

async function sendToArduino(char) {
    if (!isArduinoConnected || !serialWriter) return;
    try {
        await serialWriter.write(new TextEncoder().encode(char));
    } catch (err) {
        console.error("Serial write error:", err);
        isArduinoConnected = false;
        updateArduinoUI(false);
    }
}

async function readArduinoResponses() {
    if (!serialPort || !serialPort.readable) return;
    const decoder = new TextDecoderStream();
    const readableStreamClosed = serialPort.readable.pipeTo(decoder.writable);
    const reader = decoder.readable.getReader();
    try {
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            if (value) console.log("[Arduino]:", value.trim());
        }
    } catch (err) {
        console.log("Arduino read stream ended:", err.message);
    } finally {
        reader.releaseLock();
    }
}

function updateArduinoUI(connected) {
    const btnIcon = document.getElementById("arduinoBtnIcon");
    if (connected) {
        btnArduino.classList.add("connected");
        arduinoBtnText.textContent = "Disconnect Serial";
        if (btnIcon) btnIcon.className = "fas fa-plug-circle-check";
        arduinoDot.classList.add("connected");
        arduinoStatusText.textContent = "Arduino: Connected (Serial Active)";
        arduinoStatusBar.classList.add("connected");
    } else {
        btnArduino.classList.remove("connected");
        arduinoBtnText.textContent = "Connect Arduino";
        if (btnIcon) btnIcon.className = "fas fa-plug";
        arduinoDot.classList.remove("connected");
        arduinoStatusText.textContent = "Arduino: Disconnected";
        arduinoStatusBar.classList.remove("connected");
    }
}

// ═══════════════════════════════════════════════════════════
//  5. MODE SWITCHING
// ═══════════════════════════════════════════════════════════
function switchMode(mode) {
    if (mode === "live") {
        liveModeContainer.style.display = "";
        imageModeContainer.style.display = "none";
        btnModeLive.classList.add("active");
        btnModeImage.classList.remove("active");
    } else {
        if (isLiveRunning) stopLiveVideo();
        liveModeContainer.style.display = "none";
        imageModeContainer.style.display = "";
        btnModeImage.classList.add("active");
        btnModeLive.classList.remove("active");
    }
}

btnModeLive.addEventListener("click", () => switchMode("live"));
btnModeImage.addEventListener("click", () => switchMode("image"));

// ═══════════════════════════════════════════════════════════
//  6. IMAGE UPLOAD MODE (original functionality preserved)
// ═══════════════════════════════════════════════════════════

function drawResults(imgElement, detections) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const w = imgElement.naturalWidth || imgElement.width;
    const h = imgElement.naturalHeight || imgElement.height;
    canvas.width = w;
    canvas.height = h;

    ctx.drawImage(imgElement, 0, 0, w, h);

    detections.forEach((det) => {
        const color = CLASS_COLORS[det.className] || "#ffffff";
        const bx = det.x1, by = det.y1;
        const bw = det.x2 - det.x1, bh = det.y2 - det.y1;

        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, Math.round(w / 150));
        ctx.strokeRect(bx, by, bw, bh);

        const label = `${det.className.replace("_", " ")} ${Math.round(det.confidence * 100)}%`;
        const fontSize = Math.max(12, Math.round(w / 30));
        ctx.font = `bold ${fontSize}px Inter, sans-serif`;
        const textW = ctx.measureText(label).width;
        const labelH = fontSize + 8;

        ctx.fillStyle = color;
        ctx.fillRect(bx, by - labelH, textW + 12, labelH);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, bx + 6, by - 5);
    });

    return canvas;
}

function buildResultCard(filename, canvas, detections) {
    const card = document.createElement("div");
    card.className = "card result-card";

    const helmetCount = detections.filter((d) => d.className === "helmet").length;
    const noHelmetCount = detections.filter((d) => d.className === "no_helmet").length;

    let summaryText, summaryClass;
    if (detections.length === 0) {
        summaryText = "⚠️ No Detection — Camera blocked or no person";
        summaryClass = "yellow";
    } else if (helmetCount > 0 && noHelmetCount === 0) {
        summaryText = "✅ Helmet Detected — Ignition ENABLED";
        summaryClass = "green";
    } else if (noHelmetCount > 0 && helmetCount === 0) {
        summaryText = "❌ No Helmet — Ignition BLOCKED";
        summaryClass = "red";
    } else {
        summaryText = "⚠️ Mixed — Helmet + No-helmet detected";
        summaryClass = "orange";
    }

    // Send to Arduino based on image result
    if (helmetCount > 0 && noHelmetCount === 0) {
        sendToArduino("H");
    } else {
        sendToArduino("N");
    }

    let detHTML = "";
    if (detections.length === 0) {
        detHTML = `<div class="det-item"><span class="dn"><span class="dd" style="background:var(--yellow)"></span>⚠️ No objects</span><span class="dc medium">0%</span></div>`;
    } else {
        detections.forEach((d) => {
            const pct = Math.round(d.confidence * 100);
            const confClass = pct >= 80 ? "high" : pct >= 50 ? "medium" : "low";
            const emoji = d.className === "helmet" ? "🪖" : "⛑️";
            detHTML += `<div class="det-item"><span class="dn"><span class="dd ${d.className}"></span>${emoji} ${d.className.replace("_", " ")}</span><span class="dc ${confClass}">${pct}%</span></div>`;
        });
    }

    card.innerHTML = `
        <div class="result-filename">📄 ${filename}</div>
        <div class="status-banner ${summaryClass}">${summaryText}</div>
        <div class="result-canvas-wrap"></div>
        <div class="stats-row">
            <div class="stat-box"><div class="sv">${detections.length}</div><div class="sl">Detections</div></div>
            <div class="stat-box"><div class="sv" style="color:var(--green)">${helmetCount}</div><div class="sl">Helmet</div></div>
            <div class="stat-box"><div class="sv" style="color:var(--red)">${noHelmetCount}</div><div class="sl">No Helmet</div></div>
        </div>
        ${detHTML}
    `;

    card.querySelector(".result-canvas-wrap").appendChild(canvas);
    return card;
}

async function processAllFiles() {
    if (!session || currentFiles.length === 0) return;

    btnDetect.disabled = true;
    processingBar.style.display = "flex";
    placeholder.style.display = "none";
    resultsContainer.innerHTML = "";

    for (let i = 0; i < currentFiles.length; i++) {
        processingText.textContent = `Processing ${i + 1} of ${currentFiles.length}...`;
        const file = currentFiles[i];
        const img = await loadImageFromFile(file);
        const detections = await runInference(img);
        const canvas = drawResults(img, detections);
        const card = buildResultCard(file.name, canvas, detections);
        resultsContainer.appendChild(card);
    }

    processingBar.style.display = "none";
    btnDetect.disabled = false;
}

function loadImageFromFile(file) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = URL.createObjectURL(file);
    });
}

// ═══════════════════════════════════════════════════════════
//  7. EVENT LISTENERS
// ═══════════════════════════════════════════════════════════

// Live Video
btnStartLive.addEventListener("click", startLiveVideo);
btnStopLive.addEventListener("click", stopLiveVideo);

// Arduino
btnArduino.addEventListener("click", connectArduino);

// Image Upload
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFiles(fileInput.files);
});

function handleFiles(files) {
    currentFiles = Array.from(files);
    fileCountBadge.textContent = currentFiles.length + (currentFiles.length === 1 ? " image selected" : " images selected");
    fileCountBadge.style.display = "inline-block";
    btnDetect.disabled = false;
    if (webcamStream) closeWebcam();
}

btnDetect.addEventListener("click", processAllFiles);

// Single-shot Webcam (image mode)
btnWebcam.addEventListener("click", async () => {
    if (webcamStream) { closeWebcam(); return; }
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 640, height: 480 } });
        webcamVideo.srcObject = webcamStream;
        webcamBox.style.display = "block";
    } catch (err) {
        alert("Webcam access denied: " + err.message);
    }
});

btnCloseWc.addEventListener("click", closeWebcam);

function closeWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach((t) => t.stop());
        webcamStream = null;
    }
    webcamVideo.srcObject = null;
    webcamBox.style.display = "none";
}

btnCapture.addEventListener("click", async () => {
    if (!webcamStream) return;
    const c = document.createElement("canvas");
    c.width = webcamVideo.videoWidth;
    c.height = webcamVideo.videoHeight;
    c.getContext("2d").drawImage(webcamVideo, 0, 0);
    c.toBlob((blob) => {
        const file = new File([blob], "webcam.jpg", { type: "image/jpeg" });
        handleFiles([file]);
        processAllFiles();
    }, "image/jpeg", 0.9);
});

// ═══════════════════════════════════════════════════════════
//  8. INIT
// ═══════════════════════════════════════════════════════════
loadModel();
