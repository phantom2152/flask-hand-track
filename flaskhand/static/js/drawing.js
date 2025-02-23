// Initialize Bootstrap components
const toast = new bootstrap.Toast(document.getElementById('successToast'));

// Initialize canvas and video elements
const video = document.getElementById("videoElement");
const drawingCanvas = document.getElementById("drawingCanvas");
const indicatorCanvas = document.getElementById("indicatorCanvas");
const drawingCtx = drawingCanvas.getContext("2d");
const indicatorCtx = indicatorCanvas.getContext("2d");

// Set canvas dimensions
drawingCanvas.width = 640;
drawingCanvas.height = 480;
indicatorCanvas.width = 640;
indicatorCanvas.height = 480;

// Drawing state
let isDrawing = false;
let prevPoint = null;
let frameProcessing = false;
let isConnected = false;

// Socket connection
const socket = io();

// Camera setup
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        video.srcObject = stream;
        return true;
    } catch (error) {
        console.error("Error accessing camera:", error);
        showToast("Could not access camera. Please check permissions.", "danger");
        return false;
    }
}

// Frame capture and processing
function captureFrame() {
    if (!isConnected || frameProcessing) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    frameProcessing = true;
    socket.emit("process_frame", {
        frame: canvas.toDataURL("image/jpeg", 0.7)
    });
}

// Hand tracking and drawing functions
function handleHandTracking(handData) {
    indicatorCtx.clearRect(0, 0, indicatorCanvas.width, indicatorCanvas.height);

    if (handData.has_hand) {
        const distance = calculateDistance(handData.thumb_pos, handData.index_pos);
        const minDistance = parseInt(document.getElementById("minDistance").value);

        if (distance < minDistance) {
            // Draw indicator circle
            const midX = (handData.thumb_pos[0] + handData.index_pos[0]) / 2;
            const midY = (handData.thumb_pos[1] + handData.index_pos[1]) / 2;

            indicatorCtx.beginPath();
            indicatorCtx.arc(midX, midY, 15, 0, Math.PI * 2);
            indicatorCtx.strokeStyle = document.getElementById("drawingColor").value;
            indicatorCtx.lineWidth = 2;
            indicatorCtx.stroke();

            // Handle drawing
            if (!isDrawing) {
                isDrawing = true;
                prevPoint = handData.index_pos;
            } else if (prevPoint) {
                drawLine(prevPoint, handData.index_pos);
            }
            prevPoint = handData.index_pos;
        } else {
            isDrawing = false;
            prevPoint = null;
        }
    }
}

function calculateDistance(p1, p2) {
    if (!p1 || !p2) return Infinity;
    return Math.sqrt(Math.pow(p1[0] - p2[0], 2) + Math.pow(p1[1] - p2[1], 2));
}

function drawLine(start, end) {
    drawingCtx.beginPath();
    drawingCtx.moveTo(start[0], start[1]);
    drawingCtx.lineTo(end[0], end[1]);
    drawingCtx.strokeStyle = document.getElementById("drawingColor").value;
    drawingCtx.lineWidth = document.getElementById("lineThickness").value;
    drawingCtx.lineCap = "round";
    drawingCtx.stroke();
}

// UI Helpers
function showToast(message, type = "success") {
    const toastEl = document.getElementById('successToast');
    toastEl.querySelector('.toast-body').textContent = message;
    toastEl.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.show();
}

function updateValueDisplays() {
    ["minDistance", "lineThickness"].forEach(id => {
        const element = document.getElementById(id + "Value");
        if (element) {
            element.textContent = document.getElementById(id).value;
        }
    });
}

// Socket event handlers
socket.on("connect", () => {
    console.log("Connected to server");
    isConnected = true;
    setupCamera();
});

socket.on("disconnect", () => {
    console.log("Disconnected from server");
    isConnected = false;
    showToast("Connection lost. Reconnecting...", "danger");
});

socket.on("frame_processed", (data) => {
    frameProcessing = false;
    if (data.hand_data) {
        handleHandTracking(data.hand_data);
    }
    requestAnimationFrame(captureFrame);
});

socket.on("drawing_saved", (response) => {
    if (response.status === "success") {
        showToast("Drawing saved successfully!");
        window.location.href = `/drawings/${response.drawing_id}`;
    } else {
        showToast(response.message || "Error saving drawing", "danger");
    }
});

// Event listeners
document.getElementById("clearCanvas").addEventListener("click", () => {
    drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
});

document.getElementById("saveDrawing").addEventListener("click", () => {
    const drawingData = drawingCanvas.toDataURL("image/png");
    socket.emit("save_drawing", { image: drawingData });
    showToast("Saving drawing...");
});

// Range input listeners
["minDistance", "lineThickness"].forEach(id => {
    const element = document.getElementById(id);
    if (element) {
        element.addEventListener("input", updateValueDisplays);
    }
});

// Start video feed
video.addEventListener("play", () => {
    captureFrame();
});

// Initialize values
updateValueDisplays();