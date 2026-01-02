import cv2
import numpy as np
import os
import pickle
from openvino.runtime import Core

# =============================
# CONFIGURATION
# =============================
STABLE_SECONDS = 3
IOU_THRESHOLD = 0.3
EMBEDDING_THRESHOLD = 0.68
MAX_MISSING = 1
IS_ENTRY_CAMERA = True

EMBEDDING_FILE = "embeddings_store.pkl"

# =============================
# GLOBAL STATE
# =============================
tracks = {}
lost_tracks = {}
potential_tracks = {}
embedding_db = {}

next_track_id = 0
next_temp_id = 0

# =============================
# PERSISTENT STORAGE
# =============================
def load_embedding_store():
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_embedding_store(store):
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(store, f)

# =============================
# MODEL LOADING
# =============================
def load_models():
    ie = Core()
    return {
        "face_det": ie.compile_model(
            "models/face-detection-retail-0004/face-detection-retail-0004.xml", "CPU"),
        "body_det": ie.compile_model(
            "models/person-detection-retail-0013/person-detection-retail-0013.xml", "CPU"),
        "reid": ie.compile_model(
            "models/face-reidentification-retail-0095/face-reidentification-retail-0095.xml", "CPU"),
    }

# =============================
# UTILITIES
# =============================
def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[0] + a[2], b[0] + b[2])
    yB = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    return inter / (a[2]*a[3] + b[2]*b[3] - inter + 1e-6)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =============================
# FACE PIPELINE
# =============================
def detect_faces(frame, model):
    h, w = frame.shape[:2]
    blob = cv2.resize(frame, (300, 300)).transpose(2, 0, 1)[None].astype(np.float32)
    out = model({model.inputs[0].any_name: blob})[model.outputs[0].any_name]
    faces = []
    for d in out[0][0]:
        if d[2] > 0.5:
            faces.append((int(d[3]*w), int(d[4]*h),
                          int(d[5]*w), int(d[6]*h)))
    return faces

def extract_embedding(face_crop, reid_model):
    blob = cv2.resize(face_crop, (128, 128)).transpose(2, 0, 1)[None].astype(np.float32)
    return reid_model({reid_model.inputs[0].any_name: blob})[
        reid_model.outputs[0].any_name].flatten()

# =============================
# ID RESOLUTION (SAFE)
# =============================
def resolve_existing_id(emb, box):
    store = load_embedding_store()

    for tid, t in tracks.items():
        if cosine(emb, t["emb"]) > EMBEDDING_THRESHOLD:
            return tid

    for tid, t in lost_tracks.items():
        if cosine(emb, t["emb"]) > EMBEDDING_THRESHOLD:
            tracks[tid] = {
                "id": tid, "bbox": box, "emb": t["emb"],
                "miss": 0, "src": "face"
            }
            embedding_db[tid] = t["emb"]
            del lost_tracks[tid]
            return tid

    for tid, e in store.items():
        if cosine(emb, e) > EMBEDDING_THRESHOLD:
            tracks[tid] = {
                "id": tid, "bbox": box, "emb": e,
                "miss": 0, "src": "face"
            }
            embedding_db[tid] = e
            return tid

    return None

# =============================
# STABLE ENTRY-ONLY ID ASSIGNMENT
# =============================
def promote_stable_face(box, emb, stable_frames):
    global next_track_id, next_temp_id

    if not IS_ENTRY_CAMERA:
        return None

    for pid, p in potential_tracks.items():
        if iou(box, p["bbox"]) > IOU_THRESHOLD:
            p["bbox"], p["emb"], p["frames"] = box, emb, p["frames"] + 1
            break
    else:
        pid = next_temp_id
        next_temp_id += 1
        potential_tracks[pid] = {"bbox": box, "emb": emb, "frames": 1}

    if potential_tracks[pid]["frames"] < stable_frames:
        return None

    emb = potential_tracks[pid]["emb"]
    del potential_tracks[pid]

    store = load_embedding_store()
    for tid, e in store.items():
        if cosine(emb, e) > EMBEDDING_THRESHOLD:
            return tid

    tid = next_track_id
    next_track_id += 1

    tracks[tid] = {"id": tid, "bbox": box, "emb": emb, "miss": 0, "src": "face"}
    embedding_db[tid] = emb
    store[tid] = emb
    save_embedding_store(store)

    return tid

# =============================
# BODY FALLBACK
# =============================
def detect_bodies(frame, model):
    h, w = frame.shape[:2]
    blob = cv2.resize(frame, (544, 320)).transpose(2, 0, 1)[None].astype(np.float32)
    out = model({model.inputs[0].any_name: blob})[model.outputs[0].any_name]
    boxes = []
    for d in out[0][0]:
        if d[2] > 0.5:
            boxes.append((int(d[3]*w), int(d[4]*h),
                          int((d[5]-d[3])*w), int((d[6]-d[4])*h)))
    return boxes

def body_fallback(frame, body_model, active_ids):
    bodies = detect_bodies(frame, body_model)
    for tid in set(tracks.keys()) - active_ids:
        best, score = None, 0
        for b in bodies:
            s = iou(b, tracks[tid]["bbox"])
            if s > score:
                best, score = b, s
        if best:
            tracks[tid]["bbox"] = best
            tracks[tid]["src"] = "body"
            tracks[tid]["miss"] = 0
            active_ids.add(tid)

# =============================
# MAINTENANCE
# =============================
def update_missing(active_ids):
    for tid in list(tracks.keys()):
        if tid not in active_ids:
            tracks[tid]["miss"] += 1
            if tracks[tid]["miss"] > MAX_MISSING:
                lost_tracks[tid] = tracks[tid]
                del tracks[tid]

# =============================
# VISUALIZATION
# =============================
def draw(frame):
    for t in tracks.values():
        x, y, w, h = t["bbox"]
        color = (0,255,0) if t["src"] == "face" else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"ID {t['id']}", (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# =============================
# MAIN LOOP
# =============================
def main():
    models = load_models()
    cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 10
    stable_frames = int(fps * STABLE_SECONDS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        active_ids = set()
        faces = detect_faces(frame, models["face_det"])

        for x1, y1, x2, y2 in faces:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            emb = extract_embedding(crop, models["reid"])
            box = (x1, y1, x2-x1, y2-y1)

            tid = resolve_existing_id(emb, box) or \
                  promote_stable_face(box, emb, stable_frames)

            if tid is not None:
                active_ids.add(tid)

        body_fallback(frame, models["body_det"], active_ids)
        update_missing(active_ids)
        draw(frame)

        cv2.imshow("Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
