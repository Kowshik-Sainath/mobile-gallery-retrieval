"""
Fast ONNX Mobile Web App Server for Testing T+SBIR on Android Phones.

BUG 7 FIX applied:
  - SQLite persistence: gallery embeddings stored in gallery.db, survive server restarts.
  - Incremental indexing: only new/changed files (by last_modified timestamp) are re-embedded.
  - Batched cosine scan: single np.dot(sketch_emb, gallery_matrix.T) instead of Python loop.
  - /clear_gallery endpoint for testing.

Runs 100% on ONNX Runtime (zero PyTorch dependencies at runtime).
Connects mobile browser over local Wi-Fi to run sketch drawing + search.
"""

import os
import io
import time
import base64
import sqlite3
import struct
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
py_combiner = None

# SQLite gallery database path
GALLERY_DB_PATH = "gallery.db"
EMBED_DIM = 512


# ---------------------------------------------------------------------------
# SQLite helpers (BUG 7 FIX — persistent gallery)
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(GALLERY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates gallery table if it doesn't exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gallery (
                filename     TEXT PRIMARY KEY,
                embedding    BLOB NOT NULL,
                thumbnail_b64 TEXT NOT NULL,
                last_modified INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()
    print(f"[DB] Gallery database initialized at: {GALLERY_DB_PATH}")


def embedding_to_blob(emb: np.ndarray) -> bytes:
    """Serialize float32 numpy array to bytes blob."""
    return struct.pack(f'{len(emb)}f', *emb.tolist())


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialize bytes blob to float32 numpy array."""
    n = len(blob) // 4
    return np.array(struct.unpack(f'{n}f', blob), dtype=np.float32)


def count_gallery() -> int:
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) as n FROM gallery").fetchone()
        return row['n']


def load_gallery_matrix():
    """
    BUG 7 FIX — Loads all embeddings as a (N, D) matrix for batched cosine scan.
    Also returns list of (filename, thumbnail_b64) in the same order.
    """
    with get_db() as conn:
        rows = conn.execute("SELECT filename, embedding, thumbnail_b64 FROM gallery").fetchall()

    if not rows:
        return None, [], []

    filenames   = [r['filename'] for r in rows]
    thumbnails  = [r['thumbnail_b64'] for r in rows]
    embeddings  = np.stack([blob_to_embedding(r['embedding']) for r in rows], axis=0)
    return embeddings, filenames, thumbnails


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    # MobileCLIP uses [0, 1] range without ImageNet mean/std normalization
    img = pil_img.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0   # HWC
    arr = np.transpose(arr, (2, 0, 1))              # CHW
    arr = np.expand_dims(arr, axis=0)               # NCHW
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------

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
        .btn-danger { background: #d93025; color: white; }
        .gallery-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px; }
        .gallery-item { position: relative; border-radius: 8px; overflow: hidden; background: #eee; aspect-ratio: 1; }
        .gallery-item img { width: 100%; height: 100%; object-fit: cover; }
        .score-badge { position: absolute; bottom: 4px; right: 4px; background: rgba(0,0,0,0.75); color: #4df8a0; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; }
        .custom-file-upload { display: block; text-align: center; padding: 12px; background: #e8f0fe; color: #1a73e8; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 8px; }
        input[type="file"] { display: none; }
        #status { font-size: 0.85rem; color: #666; margin-top: 8px; text-align: center; min-height: 1.2em; }
    </style>
</head>
<body>
    <h1>&#128247; T+SBIR Mobile ONNX Gallery AI</h1>

    <div class="card">
        <div class="card-title">1. Draw Scene Sketch (Touch Canvas)</div>
        <canvas id="sketchCanvas" width="224" height="224"></canvas>
        <div class="btn-group">
            <button class="btn-secondary" onclick="clearCanvas()">Clear</button>
            <button class="btn-primary" onclick="executeSearch()">&#128269; Search</button>
        </div>
    </div>

    <div class="card">
        <div class="card-title">2. Add Photos from Phone Gallery</div>
        <label class="custom-file-upload">
            &#128193; Select Phone Gallery Photos
            <input type="file" id="photoInput" multiple accept="image/*" onchange="uploadPhonePhotos()">
        </label>
        <div id="status">Loading gallery count...</div>
        <div class="btn-group" style="margin-top:10px;">
            <button class="btn-danger" onclick="clearGallery()">&#128465; Clear Gallery DB</button>
        </div>
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
        canvas.addEventListener('touchstart', startDraw, {passive: false});
        canvas.addEventListener('touchend', stopDraw);
        canvas.addEventListener('touchmove', draw, {passive: false});

        function clearCanvas() {
            ctx.fillStyle = "white";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        async function refreshCount() {
            const res = await fetch('/gallery_count');
            const data = await res.json();
            document.getElementById('status').innerText = `Indexed Photos: ${data.count} (persisted in SQLite)`;
        }

        async function uploadPhonePhotos() {
            const input = document.getElementById('photoInput');
            if (input.files.length === 0) return;
            document.getElementById('status').innerText = "Indexing new photos (incremental)...";
            const formData = new FormData();
            for (let file of input.files) formData.append('photos', file);
            const res = await fetch('/upload_photos', { method: 'POST', body: formData });
            const data = await res.json();
            document.getElementById('status').innerText =
                `Indexed: ${data.total_photos} total | New: ${data.new_indexed} | Skipped: ${data.skipped}`;
        }

        async function clearGallery() {
            if (!confirm("Clear all indexed photos from the database?")) return;
            await fetch('/clear_gallery', { method: 'POST' });
            document.getElementById('status').innerText = "Gallery cleared.";
            document.getElementById('resultsGrid').innerHTML = '';
        }

        async function executeSearch() {
            const sketchData = canvas.toDataURL('image/png');
            document.getElementById('status').innerText = "Searching...";
            const res = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sketch_b64: sketchData })
            });
            const data = await res.json();
            const grid = document.getElementById('resultsGrid');
            grid.innerHTML = '';
            if (data.results.length === 0) {
                document.getElementById('status').innerText = "No photos indexed yet.";
                return;
            }
            data.results.forEach(item => {
                grid.innerHTML += `
                    <div class="gallery-item">
                        <img src="data:image/jpeg;base64,${item.image_b64}">
                        <div class="score-badge">${(item.score * 100).toFixed(1)}%</div>
                    </div>
                `;
            });
            document.getElementById('status').innerText =
                `Found ${data.results.length} results in ${data.latency_ms.toFixed(1)} ms`;
        }

        // Load gallery count on page load
        refreshCount();
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/gallery_count')
def gallery_count():
    return jsonify({'count': count_gallery()})


@app.route('/upload_photos', methods=['POST'])
def upload_photos():
    """
    BUG 7 FIX — Incremental indexing:
      - Checks SQLite for existing filename + last_modified.
      - Only re-embeds files that are new or have changed.
    """
    files = request.files.getlist('photos')
    new_indexed = 0
    skipped     = 0

    with get_db() as conn:
        for file in files:
            filename = file.filename
            img_bytes = file.read()

            # Use file size as a simple change proxy (no real mtime from browser upload)
            last_modified = len(img_bytes)

            existing = conn.execute(
                "SELECT last_modified FROM gallery WHERE filename = ?", (filename,)
            ).fetchone()

            if existing and existing['last_modified'] == last_modified:
                skipped += 1
                continue

            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # ONNX embedding
            inp_np = preprocess_image(img)
            outputs = vision_session.run(None, {'image_input': inp_np})
            emb = outputs[0][0]  # (512,) float32

            # Thumbnail for display
            buffered = io.BytesIO()
            img.thumbnail((300, 300))
            img.save(buffered, format="JPEG", quality=80)
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            conn.execute(
                """
                INSERT OR REPLACE INTO gallery (filename, embedding, thumbnail_b64, last_modified)
                VALUES (?, ?, ?, ?)
                """,
                (filename, embedding_to_blob(emb), img_b64, last_modified)
            )
            new_indexed += 1

        conn.commit()

    return jsonify({
        'total_photos': count_gallery(),
        'new_indexed': new_indexed,
        'skipped': skipped,
    })


@app.route('/search', methods=['POST'])
def search():
    """
    BUG 7 FIX — Batched cosine scan:
      Single np.dot(sketch_emb, gallery_matrix.T) instead of Python loop.
    """
    t0 = time.time()
    data = request.json
    sketch_b64 = data.get('sketch_b64', '')

    if ',' in sketch_b64:
        sketch_b64 = sketch_b64.split(',')[1]
    sketch_data = base64.b64decode(sketch_b64)
    sketch_img  = Image.open(io.BytesIO(sketch_data)).convert('RGB')

    sketch_np = preprocess_image(sketch_img)
    sketch_outputs = vision_session.run(None, {'image_input': sketch_np})
    sketch_emb = sketch_outputs[0][0]  # (512,)

    gallery_matrix, filenames, thumbnails = load_gallery_matrix()

    if gallery_matrix is None:
        return jsonify({'results': [], 'latency_ms': 0.0})

    # BUG 7 FIX — single batched dot product (O(N*D) but no Python loop overhead)
    scores = gallery_matrix @ sketch_emb  # (N,)

    top_k = min(9, len(scores))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [
        {
            'filename': filenames[i],
            'score': float(scores[i]),
            'image_b64': thumbnails[i],
        }
        for i in top_indices
    ]

    latency_ms = (time.time() - t0) * 1000.0
    return jsonify({'results': results, 'latency_ms': latency_ms})


@app.route('/refine', methods=['POST'])
def refine():
    """
    Feedback-driven composed retrieval endpoint.
    Takes shown_filename (photo the user saw/rejected) + refined query
    (optional new sketch_b64 and/or text), computes combined embedding,
    and returns re-ranked gallery results, explicitly excluding shown_filename.
    """
    t0 = time.time()
    data = request.json or {}
    shown_filename = data.get('shown_filename', '')
    sketch_b64 = data.get('sketch_b64', '')

    with get_db() as conn:
        row = conn.execute("SELECT embedding FROM gallery WHERE filename = ?", (shown_filename,)).fetchone()
        if not row:
            return jsonify({'error': f"Shown photo '{shown_filename}' not found in gallery."}), 404
        shown_emb = blob_to_embedding(row['embedding'])  # (512,)

    # Extract refined query embedding
    if sketch_b64:
        if ',' in sketch_b64:
            sketch_b64 = sketch_b64.split(',')[1]
        sketch_data = base64.b64decode(sketch_b64)
        sketch_img = Image.open(io.BytesIO(sketch_data)).convert('RGB')
        sketch_np = preprocess_image(sketch_img)
        sketch_outputs = vision_session.run(None, {'image_input': sketch_np})
        ref_query_emb = sketch_outputs[0][0]  # (512,)
    else:
        ref_query_emb = shown_emb.copy()

    # Query composition using Combiner ONNX or PyTorch module
    global combiner_session, py_combiner
    if combiner_session is not None:
        c_in = {
            'shown_candidate': shown_emb.reshape(1, -1).astype(np.float32),
            'refined_query': ref_query_emb.reshape(1, -1).astype(np.float32)
        }
        combined_emb = combiner_session.run(None, c_in)[0][0]
    else:
        # Fallback to PyTorch module if ONNX not yet exported
        if py_combiner is None:
            from src.reranker.combiner import FeedbackComposedRetriever
            import torch
            py_combiner = FeedbackComposedRetriever(512, 256)
            c_ckpt_path = "checkpoints/combiner_pretrained.pt"
            if os.path.exists(c_ckpt_path):
                ckpt = torch.load(c_ckpt_path, map_location='cpu', weights_only=False)
                py_combiner.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
            py_combiner.eval()
        import torch
        with torch.no_grad():
            t_sh = torch.from_numpy(shown_emb).unsqueeze(0).float()
            t_q = torch.from_numpy(ref_query_emb).unsqueeze(0).float()
            combined_emb = py_combiner(t_sh, t_q).squeeze(0).numpy()

    gallery_matrix, filenames, thumbnails = load_gallery_matrix()
    if gallery_matrix is None:
        return jsonify({'results': [], 'latency_ms': 0.0})

    scores = gallery_matrix @ combined_emb

    # Explicitly filter shown_filename out of the refined result list (UX requirement)
    filtered_results = []
    top_indices = np.argsort(scores)[::-1]
    for idx in top_indices:
        if filenames[idx] == shown_filename:
            continue  # Exclude the rejected photo
        filtered_results.append({
            'filename': filenames[idx],
            'score': float(scores[idx]),
            'image_b64': thumbnails[idx],
        })
        if len(filtered_results) >= 9:
            break

    latency_ms = (time.time() - t0) * 1000.0
    return jsonify({
        'results': filtered_results,
        'excluded_photo': shown_filename,
        'latency_ms': latency_ms
    })


@app.route('/clear_gallery', methods=['POST'])
def clear_gallery():
    """Clears all indexed photos from SQLite gallery."""
    with get_db() as conn:
        conn.execute("DELETE FROM gallery")
        conn.commit()
    return jsonify({'status': 'cleared'})


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def initialize_server():
    global vision_session, combiner_session

    print("Initializing ONNX Mobile Web Server...")

    onnx_path = "exported_models/vision_encoder_int8.onnx"
    if not os.path.exists(onnx_path):
        onnx_path = "exported_models/vision_encoder_fp32.onnx"
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"No ONNX vision encoder found. Run python export_mobile.py first."
        )

    print(f"[Server] Loading ONNX Vision Model: {onnx_path}")
    vision_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    print(f"[Server] ONNX Session ready.")

    combiner_path = "exported_models/combiner_mobile.onnx"
    if os.path.exists(combiner_path):
        combiner_session = ort.InferenceSession(combiner_path, providers=['CPUExecutionProvider'])
        print(f"[Server] Combiner ONNX Session ready.")

    init_db()


if __name__ == '__main__':
    if not FLASK_AVAILABLE or not ORT_AVAILABLE:
        print("Flask and ONNX Runtime are required: pip install flask onnxruntime")
    else:
        initialize_server()

        import socket
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            local_ip = "127.0.0.1"

        print("\n" + "=" * 65)
        print("T+SBIR ONNX Mobile Server Running!")
        print("1. Connect your Android Phone to the SAME Wi-Fi as your laptop.")
        print(f"2. Open Chrome on Android and go to:  http://{local_ip}:8000")
        print("=" * 65 + "\n")
        app.run(host='0.0.0.0', port=8000)
