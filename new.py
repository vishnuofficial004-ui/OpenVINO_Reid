import cv2
import numpy as np
import pickle
import os
from datetime import datetime
from openvino.runtime import Core

# ================= CONFIG =================
STABLE_SECONDS = 1.5
IOU_THRESHOLD = 0.25
ENTRY_THRESHOLD = 0.78
SECONDARY_THRESHOLD = 0.72
MAX_MISSING = 5
MAX_EMBS = 4
BODY_IOU_FALLBACK = 0.25

EMBEDDING_FILE = "embeddings_store.pkl"

ENTRY_CAMERAS = {
    "ENTRY_1": "http://192.0.0.4:8080/video",
}

SECONDARY_CAMERAS = {
    "SEC_1": 0,
}

# ===== WINDOW LAYOUT (ONLY UI TWEAK) =====
WINDOW_W = 640
WINDOW_H = 480
WINDOW_GAP = 10
MAX_COLS = 3   # windows per row

# ================= STORAGE =================
def load_store():
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_store(store):
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(store, f)

persistent_store = load_store()
next_gid = max(persistent_store.keys(), default=-1) + 1

# ================= MODELS =================
def load_models():
    ie = Core()
    return {
        "face": ie.compile_model(
            "models/face-detection-retail-0004/face-detection-retail-0004.xml", "CPU"),
        "reid": ie.compile_model(
            "models/face-reidentification-retail-0095/face-reidentification-retail-0095.xml", "CPU"),
        "body": ie.compile_model(
            "models/person-detection-retail-0013/person-detection-retail-0013.xml", "CPU"),
    }

# ================= UTILS =================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[0] + a[2], b[0] + b[2])
    yB = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-6)

# ================= DETECTION =================
def detect_faces(frame, model):
    h, w = frame.shape[:2]
    blob = cv2.resize(frame, (300, 300)).transpose(2, 0, 1)[None].astype(np.float32)
    out = model({model.inputs[0].any_name: blob})[model.outputs[0].any_name]
    faces = []
    for d in out[0][0]:
        if d[2] > 0.5:
            x1, y1, x2, y2 = int(d[3] * w), int(d[4] * h), int(d[5] * w), int(d[6] * h)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

def detect_bodies(frame, model):
    h, w = frame.shape[:2]
    blob = cv2.resize(frame, (544, 320)).transpose(2, 0, 1)[None].astype(np.float32)
    out = model({model.inputs[0].any_name: blob})[model.outputs[0].any_name]
    bodies = []
    for d in out[0][0]:
        if d[2] > 0.5:
            x1, y1, x2, y2 = int(d[3] * w), int(d[4] * h), int(d[5] * w), int(d[6] * h)
            bodies.append((x1, y1, x2 - x1, y2 - y1))
    return bodies

def extract_embedding(crop, model):
    blob = cv2.resize(crop, (128, 128)).transpose(2, 0, 1)[None].astype(np.float32)
    emb = model({model.inputs[0].any_name: blob})[model.outputs[0].any_name].flatten()
    return emb / (np.linalg.norm(emb) + 1e-6)

# ================= CAMERA PROCESS =================
def process_camera(frame, tracks, potentials, models, is_entry, stable_frames):
    global next_gid
    active = set()

    faces = detect_faces(frame, models["face"])

    for box in faces:
        x, y, w, h = box
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        emb = extract_embedding(crop, models["reid"])

        # === TRACK UPDATE (IOU) ===
        best_tid, best_i = None, 0
        for tid, t in tracks.items():
            v = iou(box, t["bbox"])
            if v > best_i:
                best_tid, best_i = tid, v

        if best_tid is not None and best_i > IOU_THRESHOLD:
            tracks[best_tid]["bbox"] = box
            tracks[best_tid]["miss"] = 0
            active.add(best_tid)
            continue

        # === SECONDARY CAMERA RE-ID ===
        if not is_entry:
            best_gid, best_sim = None, SECONDARY_THRESHOLD
            for gid, e in persistent_store.items():
                if gid in tracks:
                    continue
                sim = cosine(emb, e)
                if sim > best_sim:
                    best_gid, best_sim = gid, sim

            if best_gid is not None:
                tracks[best_gid] = {"bbox": box, "miss": 0, "conf": best_sim}
                active.add(best_gid)
            continue

        # === ENTRY POTENTIAL ===
        pid = None
        for k, p in potentials.items():
            if iou(box, p["bbox"]) > IOU_THRESHOLD:
                pid = k
                break

        if pid is None:
            potentials[len(potentials)] = {
                "bbox": box,
                "embs": [emb],
                "frames": 1
            }
            continue

        p = potentials[pid]
        p["frames"] += 1
        p["bbox"] = box
        p["embs"].append(emb)
        p["embs"] = p["embs"][-MAX_EMBS:]

        if p["frames"] >= stable_frames:
            final_emb = np.mean(p["embs"], axis=0)
            final_emb /= np.linalg.norm(final_emb)

            best_gid, best_sim = None, ENTRY_THRESHOLD
            for gid, e in persistent_store.items():
                sim = cosine(final_emb, e)
                if sim > best_sim:
                    best_gid, best_sim = gid, sim

            if best_gid is None:
                best_gid = next_gid
                next_gid += 1
                persistent_store[best_gid] = final_emb
                save_store(persistent_store)

            tracks[best_gid] = {"bbox": box, "miss": 0, "conf": best_sim}
            active.add(best_gid)
            del potentials[pid]

    # === BODY FALLBACK ===
    lost_ids = [gid for gid in tracks if gid not in active]
    if lost_ids:
        bodies = detect_bodies(frame, models["body"])
        for gid in lost_ids:
            t = tracks[gid]
            for b in bodies:
                if iou(t["bbox"], b) > BODY_IOU_FALLBACK:
                    t["bbox"] = b
                    t["miss"] = 0
                    active.add(gid)
                    break

    # === MISS HANDLING ===
    for gid in list(tracks.keys()):
        if gid not in active:
            tracks[gid]["miss"] += 1
            if tracks[gid]["miss"] > MAX_MISSING:
                del tracks[gid]

# ================= DRAW =================
def draw_overlay(frame, tracks, color):
    for gid, t in tracks.items():
        x, y, w, h = t["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        cv2.putText(frame, f"ID {gid} {t.get('conf', 0):.2f}",
                    (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    now = datetime.now()
    h, w = frame.shape[:2]
    cv2.putText(frame, now.strftime("%H:%M:%S"),
                (w - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, now.strftime("%d-%m-%Y"),
                (w - 150, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

# ===== WINDOW POSITION HELPER =====
def move_window(name, index):
    row = index // MAX_COLS
    col = index % MAX_COLS
    x = col * (WINDOW_W + WINDOW_GAP)
    y = row * (WINDOW_H + WINDOW_GAP)
    cv2.moveWindow(name, x, y)

# ================= MAIN =================
def main():
    models = load_models()

    entry_caps = {k: cv2.VideoCapture(v) for k, v in ENTRY_CAMERAS.items()}
    sec_caps = {k: cv2.VideoCapture(v) for k, v in SECONDARY_CAMERAS.items()}

    entry_tracks = {k: {} for k in ENTRY_CAMERAS}
    entry_potentials = {k: {} for k in ENTRY_CAMERAS}
    sec_tracks = {k: {} for k in SECONDARY_CAMERAS}

    fps = next(iter(entry_caps.values())).get(cv2.CAP_PROP_FPS)
    stable_frames = max(5, int((fps if fps > 0 else 10) * STABLE_SECONDS))

    while True:
        # ENTRY CAMERAS
        for idx, (cid, cap) in enumerate(entry_caps.items()):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (WINDOW_W, WINDOW_H))
            process_camera(frame, entry_tracks[cid], entry_potentials[cid],
                           models, True, stable_frames)
            draw_overlay(frame, entry_tracks[cid], (0, 0, 255))

            cv2.imshow(cid, frame)
            move_window(cid, idx)

        # SECONDARY CAMERAS
        offset = len(entry_caps)
        for idx, (cid, cap) in enumerate(sec_caps.items()):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (WINDOW_W, WINDOW_H))
            process_camera(frame, sec_tracks[cid], {},
                           models, False, stable_frames)
            draw_overlay(frame, sec_tracks[cid], (0, 255, 0))

            cv2.imshow(cid, frame)
            move_window(cid, offset + idx)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


