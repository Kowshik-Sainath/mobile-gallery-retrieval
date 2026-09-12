"""
Fast ONNX Mobile Web App Server for Testing T+SBIR on Android Phones.
Runs 100% on ONNX Runtime (Zero PyTorch dependencies during web server runtime).
Instantly connects mobile browser over local Wi-Fi to run sketch drawing + search over phone gallery.
"""

import os
import io
import time
import base64
import numpy as np
from PIL import Image

try:
    from flask import Flask, render_template_string, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

app = Flask(__name__) if FLASK_AVAILABLE else None

# ONNX Sessions
vision_session = None
combiner_session = None

gallery_db = [] # List of {'filename': str, 'embedding': numpy_array, 'image_b64': str}

# CLIP Normalization constants
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess_image(pil_img):
    img = pil_img.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC
    arr = np.transpose(arr, (2, 0, 1))             # CHW
    arr = np.expand_dims(arr, axis=0)              # NCHW (1, 3, 224, 224)
    arr = (arr - MEAN) / STD
    return arr.astype(np.float32)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>T+SBIR Mobile Photo Gallery AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: #f4f6f9; padding: 15px; color: #333; }
        h1 { font-size: 1.3rem; text-align: center; margin-bottom: 12px; color: #1a73e8; }
        .card { background: white; border-radius: 12px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .card-title { font-size: 1rem; font-weight: 600; margin-bottom: 10px; color: #444; }
        canvas { border: 2px solid #ddd; border-radius: 8px; background: white; touch-action: none; display: block; width: 100%; height: 224px; cursor: crosshair; }
        .btn-group { display: flex; gap: 8px; margin-top: 10px; }
        button { flex: 1; padding: 12px; border: none; border-radius: 8px; font-weight: 600; font-size: 0.95rem; cursor: pointer; }
        .btn-primary { background: #1a73e8; color: white; }
        .btn-secondary { background: #e8eaed; color: #3c4043; }
        .gallery-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px; }
        .gallery-item { position: relative; border-radius: 8px; overflow: hidden; background: #eee; aspect-ratio: 1; }
        .gallery-item img { width: 100%; height: 100%; object-fit: cover; }
        .score-badge { position: absolute; bottom: 4px; right: 4px; background: rgba(0,0,0,0.75); color: #4df8a0; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; }
        .custom-file-upload { display: block; text-align: center; padding: 12px; background: #e8f0fe; color: #1a73e8; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 8px; }
        input[type="file"] { display: none; }
    </style>
</head>
<body>
    <h1>📱 T+SBIR Mobile ONNX Gallery AI</h1>

    <div class="card">
        <div class="card-title">1. Draw Scene Sketch (Touch Canvas)</div>
        <canvas id="sketchCanvas" width="224" height="224"></canvas>
        <div class="btn-group">
            <button class="btn-secondary" onclick="clearCanvas()">Clear Canvas</button>
            <button class="btn-primary" onclick="executeSearch()">🔍 Search Photos</button>
        </div>
    </div>

    <div class="card">
        <div class="card-title">2. Add Photos from Phone Gallery</div>
        <label class="custom-file-upload">
            📁 Select Phone Gallery Photos
            <input type="file" id="photoInput" multiple accept="image/*" onchange="uploadPhonePhotos()">
        </label>
        <div id="galleryCount" style="font-size:0.85rem; color:#666; margin-top:8px; text-align:center;">Indexed Photos: 0</div>
    </div>

    <div class="card">
        <div class="card-title">Search Results (ONNX INT8 Ranked)</div>
        <div class="gallery-grid" id="resultsGrid"></div>
    </div>

    <script>
        const canvas = document.getElementById('sketchCanvas');
        const ctx = canvas.getContext('2d');
        let drawing = false;

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'black';

        function getPos(e) {
            const rect = canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return {
                x: (clientX - rect.left) * (canvas.width / rect.width),
                y: (clientY - rect.top) * (canvas.height / rect.height)
            };
        }

        function startDraw(e) { drawing = true; draw(e); }
        function stopDraw() { drawing = false; ctx.beginPath(); }
        function draw(e) {
            if (!drawing) return;
            e.preventDefault();
            const pos = getPos(e);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(pos.x, pos.y);
        }

        canvas.addEventListener('mousedown', startDraw);
        canvas.addEventListener('mouseup', stopDraw);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('touchstart', startDraw);
        canvas.addEventListener('touchend', stopDraw);
        canvas.addEventListener('touchmove', draw);

        function clearCanvas() {
            ctx.fillStyle = "white";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        async function uploadPhonePhotos() {
            const input = document.getElementById('photoInput');
            if (input.files.length === 0) return;

            const formData = new FormData();
            for (let file of input.files) {
                formData.append('photos', file);
            }

            document.getElementById('galleryCount').innerText = "Indexing photos via ONNX INT8 model...";
            const res = await fetch('/upload_photos', { method: 'POST', body: formData });
            const data = await res.json();
            document.getElementById('galleryCount').innerText = `Indexed Photos: ${data.total_photos}`;
        }

        async function executeSearch() {
            const sketchData = canvas.toDataURL('image/png');

            const res = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sketch_b64: sketchData })
            });
            const data = await res.json();

            const grid = document.getElementById('resultsGrid');
            grid.innerHTML = '';
            data.results.forEach(item => {
                grid.innerHTML += `
                    <div class="gallery-item">
                        <img src="data:image/jpeg;base64,${item.image_b64}">
                        <div class="score-badge">${(item.score * 100).toFixed(1)}%</div>
                    </div>
                `;
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload_photos', methods=['POST'])
def upload_photos():
    files = request.files.getlist('photos')
    for file in files:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Preprocess & compute ONNX embedding
        inp_np = preprocess_image(img)
        outputs = vision_session.run(None, {'image_input': inp_np})
        emb = outputs[0][0] # 512-d normalized embedding

        # Thumbnail for display
        buffered = io.BytesIO()
        img.thumbnail((300, 300))
        img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        gallery_db.append({
            'filename': file.filename,
            'embedding': emb,
            'image_b64': img_b64
        })

    return jsonify({'total_photos': len(gallery_db)})

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    sketch_b64 = data.get('sketch_b64')

    sketch_data = base64.b64decode(sketch_b64.split(',')[1])
    sketch_img = Image.open(io.BytesIO(sketch_data)).convert('RGB')
    
    # Preprocess & compute sketch embedding via ONNX model
    sketch_np = preprocess_image(sketch_img)
    sketch_outputs = vision_session.run(None, {'image_input': sketch_np})
    sketch_emb = sketch_outputs[0][0]

    # Similarity scan over cached gallery photos
    results = []
    for item in gallery_db:
        sim_score = float(np.dot(sketch_emb, item['embedding']))
        results.append({
            'filename': item['filename'],
            'score': sim_score,
            'image_b64': item['image_b64']
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({'results': results[:9]})


def initialize_server():
    global vision_session, combiner_session
    print("Initializing ONNX Mobile Web Server...")
    
    onnx_path = "exported_models/vision_encoder_int8.onnx"
    if not os.path.exists(onnx_path):
        onnx_path = "exported_models/vision_encoder_fp32.onnx"

    print(f"Loading ONNX Model: {onnx_path}")
    vision_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    print("ONNX Model Session Ready!")


if __name__ == '__main__':
    if not FLASK_AVAILABLE or not ORT_AVAILABLE:
        print("Flask and ONNX Runtime are required. Please run: pip install flask onnxruntime")
    else:
        initialize_server()
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print("\n" + "="*65)
        print(f"T+SBIR ONNX Mobile Server Running!")
        print(f"1. Connect your Android Phone to the SAME Wi-Fi network as your laptop.")
        print(f"2. Open Chrome on your Android Phone and go to:")
        print(f"   http://{local_ip}:8000")
        print("="*65 + "\n")
        app.run(host='0.0.0.0', port=8000)
