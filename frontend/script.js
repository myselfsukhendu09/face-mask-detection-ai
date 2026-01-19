
const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const toggleBtn = document.getElementById('toggle-cam');
const maskCountEl = document.getElementById('mask-count');
const noMaskCountEl = document.getElementById('no-mask-count');

let isRunning = false;
let stream = null;

async function setupWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        video.srcObject = stream;
        return true;
    } catch (err) {
        console.error("Error accessing webcam: ", err);
        alert("Could not access webcam. Please ensure you have given permission.");
        return false;
    }
}

async function captureAndDetect() {
    if (!isRunning) return;

    // Draw video to a hidden canvas to get image data
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);

    // Convert to blob
    tempCanvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        try {
            const response = await fetch('http://localhost:8001/detect', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            drawDetections(data.detections);
        } catch (err) {
            console.error("Detection error: ", err);
        }

        // Repeat
        if (isRunning) {
            setTimeout(captureAndDetect, 100); // 10 FPS approx
        }
    }, 'image/jpeg', 0.8);
}

function drawDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let masks = 0;
    let noMasks = 0;

    detections.forEach(det => {
        const [x, y, w, h] = det.bbox;
        const label = det.label;
        const color = label.includes('No Mask') ? '#ef4444' : '#10b981';

        if (label.includes('No Mask')) noMasks++;
        else masks++;

        // Draw bbox
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, w, h);

        // Draw label background
        ctx.fillStyle = color;
        ctx.font = 'bold 16px Outfit';
        const labelText = `${label} (${Math.round(det.confidence * 100)}%)`;
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillRect(x, y - 25, textWidth + 10, 25);

        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(labelText, x + 5, y - 7);
    });

    maskCountEl.textContent = masks;
    noMaskCountEl.textContent = noMasks;
}

toggleBtn.addEventListener('click', async () => {
    if (!isRunning) {
        const success = await setupWebcam();
        if (success) {
            isRunning = true;
            toggleBtn.textContent = "Stop AI Analysis";
            toggleBtn.style.background = "#ef4444";

            // Wait for video metadata to set canvas size
            video.addEventListener('loadedmetadata', () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                captureAndDetect();
            });

            if (video.readyState >= 2) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                captureAndDetect();
            }
        }
    } else {
        isRunning = false;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        video.srcObject = null;
        toggleBtn.textContent = "Start AI Analysis";
        toggleBtn.style.background = "var(--primary)";
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
});
