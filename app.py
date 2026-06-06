"""
Violence Detection Web App v2
- Video playback with violence timeline
- Per-segment detection across the video

Run: python app.py
Open: http://localhost:5000
"""

import os, cv2, random, math
import numpy as np
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.models.video as video_models

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Config (must match training) ───────────────────────────────────────────
CFG = dict(IMG_SIZE=112, NUM_FRAMES=16, DROPOUT=0.5)
KMEAN = np.array([0.43216, 0.394666, 0.37645],  dtype=np.float32)
KSTD  = np.array([0.22803, 0.22145,  0.216989], dtype=np.float32)

# ✏️ Update these to your actual .pt file paths
MODEL_PATHS = {
    'R3D_18'      : r'models\R3D_18_best.pt',
    'MC3_18'      : r'models\MC3_18_best.pt',
    'R2Plus1D_18' : r'models\R2Plus1D_18_best.pt',
}
MODEL_WEIGHTS = {
    'R3D_18'      : 0.6538,
    'MC3_18'      : 0.7495,
    'R2Plus1D_18' : 0.7400,
}
THRESHOLD       = 0.521
SEGMENT_SECONDS = 2.0   # analyse every 2 seconds (matches training clip length)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


# ── Build model ────────────────────────────────────────────────────────────
def build_model(arch):
    builders = {
        'r3d_18'      : (video_models.r3d_18,      video_models.R3D_18_Weights.KINETICS400_V1),
        'mc3_18'      : (video_models.mc3_18,       video_models.MC3_18_Weights.KINETICS400_V1),
        'r2plus1d_18' : (video_models.r2plus1d_18,  video_models.R2Plus1D_18_Weights.KINETICS400_V1),
    }
    builder, weights = builders[arch]
    model = builder(weights=weights)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(CFG['DROPOUT']),
        nn.Linear(in_feats, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return model


arch_map = {
    'R3D_18'      : 'r3d_18',
    'MC3_18'      : 'mc3_18',
    'R2Plus1D_18' : 'r2plus1d_18',
}

# Load models at startup
loaded_models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        model = build_model(arch_map[name])
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model = model.to(DEVICE).eval()
        loaded_models[name] = model
        print(f'Loaded: {name}')
    else:
        print(f'Not found (skipping): {path}')

if not loaded_models:
    print('WARNING: No models loaded. Check MODEL_PATHS in app.py.')


# ── Frame extraction helpers ───────────────────────────────────────────────
def frames_to_tensor(frames_bgr):
    """Convert list of BGR frames → model input tensor (1,3,T,H,W)."""
    processed = []
    for f in frames_bgr:
        f = cv2.resize(f, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        f = (f - KMEAN) / KSTD
        processed.append(f)
    arr = np.stack(processed, axis=0)                             # (T,H,W,3)
    return torch.from_numpy(arr).permute(3,0,1,2).unsqueeze(0).float()  # (1,3,T,H,W)


@torch.no_grad()
def score_tensor(tensor):
    """Run ensemble on a single tensor, return probability 0-1."""
    if not loaded_models:
        return 0.5
    total_w, total_s = 0.0, 0.0
    for name, model in loaded_models.items():
        w = MODEL_WEIGHTS.get(name, 1.0)
        s = model(tensor.to(DEVICE)).squeeze().item()
        total_s += w * s
        total_w += w
    return total_s / total_w


# ── Main inference ─────────────────────────────────────────────────────────
def analyse_video(video_path):
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    if total < 2:
        cap.release()
        raise ValueError('Video too short or unreadable.')

    seg_frames = max(1, int(fps * SEGMENT_SECONDS))  # frames per segment
    segments   = []
    all_frames = []

    # Read all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    if not all_frames:
        raise ValueError('Could not read frames from video.')

    # Score each segment
    step = seg_frames
    i    = 0
    while i < len(all_frames):
        chunk = all_frames[i: i + step]
        if len(chunk) < 2:
            break

        # Uniformly sample NUM_FRAMES from chunk
        indices = np.linspace(0, len(chunk) - 1, CFG['NUM_FRAMES'], dtype=int)
        sampled = [chunk[j] for j in indices]
        tensor  = frames_to_tensor(sampled)
        score   = score_tensor(tensor)

        t_start = i / fps
        t_end   = min((i + step) / fps, duration)
        label   = 'VIOLENCE' if score >= THRESHOLD else 'NON-VIOLENCE'

        segments.append({
            'start': round(t_start, 2),
            'end'  : round(t_end,   2),
            'score': round(float(score), 4),
            'label': label,
        })
        i += step

    # Overall score = weighted average (weight by segment length)
    if segments:
        total_len   = sum(s['end'] - s['start'] for s in segments)
        overall     = sum(s['score'] * (s['end'] - s['start']) for s in segments) / total_len
    else:
        overall = 0.5

    label      = 'VIOLENCE' if overall >= THRESHOLD else 'NON-VIOLENCE'
    confidence = overall if overall >= THRESHOLD else 1 - overall

    return {
        'label'      : label,
        'score'      : round(float(overall), 4),
        'confidence' : round(float(confidence) * 100, 1),
        'threshold'  : THRESHOLD,
        'duration'   : round(duration, 2),
        'fps'        : round(fps, 1),
        'segments'   : segments,
        'models_used': list(loaded_models.keys()),
    }


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           models_loaded=list(loaded_models.keys()),
                           device=str(DEVICE))


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    f = request.files['video']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Unsupported format. Use: {", ".join(allowed)}'}), 400

    if not loaded_models:
        return jsonify({'error': 'No models loaded. Check MODEL_PATHS in app.py.'}), 500

    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)

    try:
        result = analyse_video(path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == '__main__':
    print(f'\nServer starting...')
    print(f'Open: http://localhost:5000\n')
    app.run(debug=False, host='0.0.0.0', port=5000)
