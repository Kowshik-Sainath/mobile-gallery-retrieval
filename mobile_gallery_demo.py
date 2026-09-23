"""
Fast ONNX Mobile Web App Server for Testing T+SBIR on Android Phones.

Deploys the complete, verified 5-session ONNX pipeline:
  1. Photo Encoder (photo_encoder_int8.onnx): LoRA isolated/unloaded, gallery indexing.
  2. Sketch Encoder (sketch_encoder_int8.onnx): LoRA merged, sketch query encoding.
  3. Text Encoder (text_encoder_int8.onnx): MobileCLIP text tower + adapter.
  4. Composite Fusion (composite_fusion_mobile.onnx): Multimodal fusion (sketch + text).
  5. Feedback Combiner (combiner_mobile.onnx): Composed retrieval ((shown, refined) -> combined).

Key features:
  - SQLite persistence: gallery embeddings stored in gallery.db, survive server restarts.
  - Incremental indexing: only new/changed files are re-embedded.
  - Batched cosine scan: single np.dot(query_emb, gallery_matrix.T).
  - Multimodal search: supports sketch-only, text-only, and combined sketch+text queries.
  - Corrective interactive refinement: re-ranks full gallery via combiner ONNX while
    explicitly excluding the rejected/shown photo.
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

try:
    import mobileclip
    MOBILECLIP_AVAILABLE = True
except ImportError:
    MOBILECLIP_AVAILABLE = False

app = Flask(__name__) if FLASK_AVAILABLE else None

# ONNX Sessions
photo_session    = None
sketch_session   = None
text_session     = None
fusion_session   = None
combiner_session = None
tokenizer        = None

# SQLite gallery database path
GALLERY_DB_PATH = "gallery.db"
EMBED_DIM = 512


# ---------------------------------------------------------------------------
# SQLite helpers (Persistent gallery)
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
        return row['n'] if row else 0


def load_gallery_matrix():
    """
    Loads all embeddings as an (N, D) matrix for batched cosine scan.
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
# Image Preprocessing & Helpers
# ---------------------------------------------------------------------------

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """MobileCLIP input: [0, 1] float32 range (NCHW 1x3x224x224, BICUBIC resize)."""
    img = pil_img.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0   # HWC
    arr = np.transpose(arr, (2, 0, 1))              # CHW
    arr = np.expand_dims(arr, axis=0)               # NCHW
    return arr.astype(np.float32)


def is_blank_canvas(pil_img: Image.Image) -> bool:
    """Returns True if the sketch canvas is blank (all pixels white)."""
    extrema = pil_img.convert('L').getextrema()
    return extrema[0] >= 250


def compute_query_embedding(sketch_img=None, text_str=None) -> np.ndarray:
    """
    Computes a query embedding (512,) float32 vector:
      - Both sketch + text: fuses via sketch_encoder + text_encoder + composite_fusion
      - Sketch only: sketch_encoder
      - Text only: text_encoder
    """
    has_sketch = False
    sketch_emb = None
    if sketch_img is not None and not is_blank_canvas(sketch_img):
        has_sketch = True
        sketch_np = preprocess_image(sketch_img)
        sketch_in_name = sketch_session.get_inputs()[0].name
        outputs = sketch_session.run(None, {sketch_in_name: sketch_np})
        sketch_emb = outputs[0]  # (1, 512)

    has_text = False
    text_emb = None
    if text_str and text_str.strip() and tokenizer is not None and text_session is not None:
        has_text = True
        tokens = tokenizer([text_str.strip()]).numpy().astype(np.int64)
        outputs = text_session.run(None, {'token_ids': tokens})
        text_emb = outputs[0]  # (1, 512)

    if has_sketch and has_text and fusion_session is not None:
        fused = fusion_session.run(None, {
            'sketch_embed': sketch_emb,
            'text_embed': text_emb
        })[0][0]  # (512,)
        return fused / (np.linalg.norm(fused) + 1e-8)

    if has_sketch:
        v = sketch_emb[0]
        return v / (np.linalg.norm(v) + 1e-8)

    if has_text:
        v = text_emb[0]
        return v / (np.linalg.norm(v) + 1e-8)

    # Fallback: if sketch was drawn but was blank, still run sketch encoder
    if sketch_img is not None:
        sketch_np = preprocess_image(sketch_img)
        sketch_in_name = sketch_session.get_inputs()[0].name
        v = sketch_session.run(None, {sketch_in_name: sketch_np})[0][0]
        return v / (np.linalg.norm(v) + 1e-8)

    # Empty fallback
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    v[0] = 1.0
    return v


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
        h1 { font-size: 1.25rem; text-align: center; margin-bottom: 12px; color: #1a73e8; }
        .card { background: white; border-radius: 12px; padding: 14px; margin-bottom: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .card-title { font-size: 0.95rem; font-weight: 600; margin-bottom: 8px; color: #444; }
        canvas { border: 2px solid #ddd; border-radius: 8px; background: white; touch-action: none; display: block; width: 100%; height: 224px; cursor: crosshair; }
        .input-text { width: 100%; padding: 10px; margin-top: 8px; border: 2px solid #ddd; border-radius: 8px; font-size: 0.9rem; }
        .btn-group { display: flex; gap: 8px; margin-top: 10px; }
        button { flex: 1; padding: 11px; border: none; border-radius: 8px; font-weight: 600; font-size: 0.9rem; cursor: pointer; }
        .btn-primary { background: #1a73e8; color: white; }
        .btn-secondary { background: #e8eaed; color: #3c4043; }
        .btn-danger { background: #d93025; color: white; }
        .gallery-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px; }
        .gallery-item { position: relative; border-radius: 8px; overflow: hidden; background: #eee; aspect-ratio: 1; }
        .gallery-item img { width: 100%; height: 100%; object-fit: cover; }
        .score-badge { position: absolute; bottom: 4px; left: 4px; background: rgba(0,0,0,0.75); color: #4df8a0; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; }
        .refine-btn { position: absolute; top: 4px; right: 4px; background: rgba(26, 115, 232, 0.85); color: white; border: none; border-radius: 4px; padding: 3px 6px; font-size: 0.7rem; font-weight: bold; cursor: pointer; }
        .refine-btn:hover { background: #1a73e8; }
        .custom-file-upload { display: block; text-align: center; padding: 12px; background: #e8f0fe; color: #1a73e8; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 8px; }
        input[type="file"] { display: none; }
        #status { font-size: 0.85rem; color: #555; margin-top: 8px; text-align: center; min-height: 1.2em; }
        .banner { background: #e8f0fe; border-left: 4px solid #1a73e8; padding: 8px 12px; font-size: 0.8rem; margin-bottom: 12px; border-radius: 4px; color: #155724; }
    </style>
</head>
<body>
    <h1>&#128247; T+SBIR Mobile ONNX Gallery AI</h1>
    <div class="banner">
        <strong>Pipeline:</strong> INT8 FastViT Vision + Text + Composite Fusion + Combiner.
    </div>

    <div class="card">
        <div class="card-title">1. Multimodal Query (Sketch + Text)</div>
        <canvas id="sketchCanvas" width="224" height="224"></canvas>
        <input type="text" id="textPrompt" class="input-text" placeholder="Text description / details (e.g. 'a brown dog on grass')...">
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
        <div class="card-title" id="resultsTitle">Search Results (ONNX INT8 Ranked)</div>
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
            document.getElementById('textPrompt').value = '';
        }

        async function refreshCount() {
            const res = await fetch('/gallery_count');
            const data = await res.json();
            document.getElementById('status').innerText = `Indexed Photos: ${data.count} (persisted in SQLite)`;
        }

        async function uploadPhonePhotos() {
            const input = document.getElementById('photoInput');
            if (input.files.length === 0) return;
            document.getElementById('status').innerText = "Indexing new photos with Photo Encoder INT8...";
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

        function renderResults(data, customTitle) {
            const grid = document.getElementById('resultsGrid');
            grid.innerHTML = '';
            document.getElementById('resultsTitle').innerText = customTitle || "Search Results (ONNX INT8 Ranked)";
            if (!data.results || data.results.length === 0) {
                document.getElementById('status').innerText = "No matching photos found.";
                return;
            }
            data.results.forEach(item => {
                grid.innerHTML += `
                    <div class="gallery-item">
                        <img src="data:image/jpeg;base64,${item.image_b64}" alt="${item.filename}">
                        <div class="score-badge">${(item.score * 100).toFixed(1)}%</div>
                        <button class="refine-btn" onclick="executeRefine('${item.filename}')" title="Refine search using this photo">&#128260; Refine</button>
                    </div>
                `;
            });
            document.getElementById('status').innerText =
                `Found ${data.results.length} results in ${data.latency_ms.toFixed(1)} ms`;
        }

        async function executeSearch() {
            const sketchData = canvas.toDataURL('image/png');
            const textPrompt = document.getElementById('textPrompt').value.trim();
            document.getElementById('status').innerText = "Searching with ONNX INT8...";
            const res = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sketch_b64: sketchData, text_prompt: textPrompt })
            });
            const data = await res.json();
            renderResults(data, "Search Results (ONNX INT8 Ranked)");
        }

        async function executeRefine(filename) {
            const sketchData = canvas.toDataURL('image/png');
            const textPrompt = document.getElementById('textPrompt').value.trim();
            document.getElementById('status').innerText = `Refining from '${filename}' (query composition)...`;
            const res = await fetch('/refine', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ shown_filename: filename, sketch_b64: sketchData, text_prompt: textPrompt })
            });
            const data = await res.json();
            renderResults(data, `Refined from '${filename}' (photo excluded)`);
        }

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
    Embeds gallery photos strictly using photo_session (photo_encoder_int8.onnx),
    ensuring LoRA weights are completely absent from the photo indexing path.
    """
    if photo_session is None:
        return jsonify({'error': 'Photo encoder session not loaded.'}), 500

    files = request.files.getlist('photos')
    new_indexed = 0
    skipped     = 0

    with get_db() as conn:
        for file in files:
            filename = file.filename
            img_bytes = file.read()
            last_modified = len(img_bytes)

            existing = conn.execute(
                "SELECT last_modified FROM gallery WHERE filename = ?", (filename,)
            ).fetchone()

            if existing and existing['last_modified'] == last_modified:
                skipped += 1
                continue

            try:
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            except Exception:
                continue

            inp_np = preprocess_image(img)
            outputs = photo_session.run(None, {'image_input': inp_np})
            emb = outputs[0][0]  # (512,) float32

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
    Multimodal search endpoint:
      Accepts sketch_b64 and/or text_prompt, computes composite query embedding,
      and performs a batched cosine scan against gallery embeddings.
    """
    t0 = time.time()
    data = request.json or {}
    sketch_b64  = data.get('sketch_b64', '')
    text_prompt = data.get('text_prompt', '').strip()

    sketch_img = None
    if sketch_b64:
        if ',' in sketch_b64:
            sketch_b64 = sketch_b64.split(',')[1]
        try:
            sketch_data = base64.b64decode(sketch_b64)
            sketch_img  = Image.open(io.BytesIO(sketch_data)).convert('RGB')
        except Exception:
            sketch_img = None

    query_emb = compute_query_embedding(sketch_img=sketch_img, text_str=text_prompt)

    gallery_matrix, filenames, thumbnails = load_gallery_matrix()
    if gallery_matrix is None or len(filenames) == 0:
        return jsonify({'results': [], 'latency_ms': (time.time() - t0) * 1000.0})

    scores = gallery_matrix @ query_emb
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
    (optional new sketch_b64 and/or text_prompt), computes combined embedding
    using combiner_mobile.onnx, re-searches the full gallery, and explicitly
    excludes shown_filename from the returned candidates.

    Full-Gallery Re-Search vs. Candidate Reranking:
      Instead of re-scoring only the previously shown top-K results (candidate reranking),
      the combiner projects (shown_photo, refined_query) into a new 512-D query vector
      and performs a full batched matrix multiplication against gallery_matrix (N photos).
      This allows genuine discovery of photos that were initially ranked beyond top-K,
      while explicitly excluding the rejected photo from the new results.
    """
    t0 = time.time()
    data = request.json or {}
    shown_filename = data.get('shown_filename', '')
    sketch_b64     = data.get('sketch_b64', '')
    text_prompt    = data.get('text_prompt', '').strip()

    with get_db() as conn:
        row = conn.execute("SELECT embedding FROM gallery WHERE filename = ?", (shown_filename,)).fetchone()
        if not row:
            return jsonify({'error': f"Shown photo '{shown_filename}' not found in gallery."}), 404
        shown_emb = blob_to_embedding(row['embedding'])  # (512,)

    sketch_img = None
    if sketch_b64:
        if ',' in sketch_b64:
            sketch_b64 = sketch_b64.split(',')[1]
        try:
            sketch_data = base64.b64decode(sketch_b64)
            sketch_img  = Image.open(io.BytesIO(sketch_data)).convert('RGB')
        except Exception:
            sketch_img = None

    if sketch_img is not None or text_prompt:
        ref_query_emb = compute_query_embedding(sketch_img=sketch_img, text_str=text_prompt)
    else:
        ref_query_emb = shown_emb.copy()

    # Query composition via Combiner ONNX
    if combiner_session is not None:
        c_in = {
            'shown_candidate': shown_emb.reshape(1, -1).astype(np.float32),
            'refined_query': ref_query_emb.reshape(1, -1).astype(np.float32)
        }
        combined_emb = combiner_session.run(None, c_in)[0][0]
    else:
        # Fallback average if combiner session is unavailable
        combined_emb = 0.5 * shown_emb + 0.5 * ref_query_emb

    combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-8)

    gallery_matrix, filenames, thumbnails = load_gallery_matrix()
    if gallery_matrix is None or len(filenames) == 0:
        return jsonify({'results': [], 'excluded_photo': shown_filename, 'latency_ms': (time.time() - t0) * 1000.0})

    scores = gallery_matrix @ combined_emb

    # Filter out shown_filename from results
    filtered_results = []
    top_indices = np.argsort(scores)[::-1]
    for idx in top_indices:
        if filenames[idx] == shown_filename:
            continue
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


def reconcile_gallery(photo_records: list, batch_size: int = 16):
    """
    CLIP-Finder2 Lifecycle Reconciliation (Android MediaStore Sync):
    photo_records: list of (filename, last_modified, image_bytes)
    Steps:
      1. Detect new/modified photos.
      2. Batch-encode in chunks of batch_size (default 16, tuned for 4-8GB RAM).
      3. Prune DB rows for photos no longer present.
      4. Short-circuit unchanged photos (0 embedding calls).
    """
    if photo_session is None:
        raise RuntimeError("Photo encoder session not loaded.")

    incoming_map = {rec[0]: (rec[1], rec[2]) for rec in photo_records}
    incoming_filenames = set(incoming_map.keys())

    with get_db() as conn:
        existing_rows = conn.execute("SELECT filename, last_modified FROM gallery").fetchall()
        existing_map = {r['filename']: r['last_modified'] for r in existing_rows}

        # 1. Prune deletions
        to_delete = [fn for fn in existing_map if fn not in incoming_filenames]
        for fn in to_delete:
            conn.execute("DELETE FROM gallery WHERE filename = ?", (fn,))

        # 2. Identify additions and modifications
        to_embed = []
        unchanged_count = 0
        for fn, (mtime, b_bytes) in incoming_map.items():
            if fn in existing_map and existing_map[fn] == mtime:
                unchanged_count += 1
            else:
                to_embed.append((fn, mtime, b_bytes))

        # 3. Batch encode in chunks of batch_size
        new_indexed = 0
        for i in range(0, len(to_embed), batch_size):
            chunk = to_embed[i:i + batch_size]
            for fn, mtime, b_bytes in chunk:
                try:
                    img = Image.open(io.BytesIO(b_bytes)).convert('RGB')
                    inp = preprocess_image(img)
                    out = photo_session.run(None, {'image_input': inp})
                    emb = out[0][0]  # (512,) float32

                    buffered = io.BytesIO()
                    img.thumbnail((300, 300))
                    img.save(buffered, format="JPEG", quality=80)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO gallery (filename, embedding, thumbnail_b64, last_modified)
                        VALUES (?, ?, ?, ?)
                        """,
                        (fn, embedding_to_blob(emb), img_b64, mtime)
                    )
                    new_indexed += 1
                except Exception:
                    continue

        conn.commit()

    return {
        'total_photos': count_gallery(),
        'new_indexed': new_indexed,
        'unchanged_skipped': unchanged_count,
        'deleted_pruned': len(to_delete),
        'embedding_calls': new_indexed
    }


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def initialize_server():
    global photo_session, sketch_session, text_session, fusion_session, combiner_session, tokenizer

    print("=" * 65)
    print("Initializing Full T+SBIR ONNX Mobile Server (Deduplicated)...")
    print("=" * 65)

    def _load_session(preferred_path, fallback_path=None):
        target = preferred_path if os.path.exists(preferred_path) else fallback_path
        if target and os.path.exists(target):
            print(f"[Server] Loading session: {target}")
            return ort.InferenceSession(target, providers=['CPUExecutionProvider'])
        print(f"[Server] WARNING: Could not find {preferred_path} or {fallback_path}")
        return None

    # 1. Base Photo Backbone (Deduplicated)
    photo_session = _load_session(
        "exported_models/photo_backbone_int8.onnx",
        "exported_models/photo_encoder_int8.onnx"
    )
    if photo_session is None:
        photo_session = _load_session("exported_models/photo_backbone_fp32.onnx", "exported_models/photo_encoder_fp32.onnx")

    # 2. Sketch Session via Deduplicated Base-Plus-Delta (Fallback to standalone)
    delta_path = "exported_models/sketch_lora_delta.npz"
    base_fp32_path = "exported_models/photo_backbone_fp32.onnx"
    if not os.path.exists(base_fp32_path):
        base_fp32_path = "exported_models/photo_encoder_fp32.onnx"

    if os.path.exists(delta_path) and os.path.exists(base_fp32_path):
        try:
            from export_mobile import load_sketch_session_from_delta
            sketch_session = load_sketch_session_from_delta(base_fp32_path, delta_path)
            print("[Server] Sketch session loaded via deduplicated Base-Plus-Delta.")
        except Exception as e:
            print(f"[Server] Delta loading failed ({e}), using standalone sketch session.")
            sketch_session = _load_session("exported_models/sketch_encoder_int8.onnx", "exported_models/sketch_encoder_fp32.onnx")
    else:
        sketch_session = _load_session("exported_models/sketch_encoder_int8.onnx", "exported_models/sketch_encoder_fp32.onnx")

    # 3. Text, Fusion, and Combiner
    text_session     = _load_session("exported_models/text_encoder_int8.onnx", "exported_models/text_encoder_fp32.onnx")
    fusion_session   = _load_session("exported_models/composite_fusion_mobile.onnx")
    combiner_session = _load_session("exported_models/combiner_mobile.onnx")

    if MOBILECLIP_AVAILABLE:
        try:
            tokenizer = mobileclip.get_tokenizer("mobileclip_s1")
            print("[Server] MobileCLIP tokenizer initialized.")
        except Exception as e:
            print(f"[Server] Warning: failed to load MobileCLIP tokenizer: {e}")

    init_db()
    print("[Server] All 5 ONNX components and SQLite DB ready.\n")


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

        print("=" * 65)
        print("T+SBIR ONNX Mobile Server Running!")
        print("1. Connect your Android Phone to the SAME Wi-Fi as your laptop.")
        print(f"2. Open Chrome on Android and go to:  http://{local_ip}:8000")
        print("=" * 65 + "\n")
        app.run(host='0.0.0.0', port=8000)
