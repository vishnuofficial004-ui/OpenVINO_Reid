import cv2
import numpy as np
import pickle
import os
import time
from openvino.runtime import Core

# ================= CONFIG =================
STABLE_SECONDS = 1.5
IOU_THRESHOLD = 0.3
ENTRY_THRESHOLD = 0.80
SECONDARY_THRESHOLD = 0.72
MAX_MISSING_SECONDS = 3.0  # CHANGED: was MAX_MISSING = 5 (frame count)

EMBEDDING_FILE = "embeddings_store.pkl"

# ---- NEW: gallery settings ----
# Instead of storing ONE averaged embedding per identity, we keep a small
# rolling set of embeddings ("gallery") captured at different times/angles.
# Matching checks similarity against the whole gallery, not one vector,
# which is far more robust over long sessions.
GALLERY_SIZE = 5          # max embeddings kept per identity
GALLERY_ADD_THRESHOLD = 0.85  # only add a new embedding if it's "different enough"

# ---- NEW: list of cameras instead of 2 hardcoded ones ----
# Each camera config declares its own source and whether it acts
# as an "entry" camera (i.e. allowed to REGISTER new identities)
# or a "secondary" camera (can only MATCH against existing ones).
CAMERAS = [
    {"name": "MOBILE_ENTRY", "source": "http://192.168.1.5:8080/video", "is_entry": True},
    {"name": "PC_SECONDARY",  "source": 0,                               "is_entry": False},
    # Add more cameras here, e.g.:
    # {"name": "HALLWAY_CAM", "source": "rtsp://192.168.1.20/stream", "is_entry": False},
]

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
            "models/face-reidentification-retail-0095/face-reidentification-retail-0095.xml", "CPU")
    }

# ================= UTILS =================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[0]+a[2], b[0]+b[2])
    yB = min(a[1]+a[3], b[1]+b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    return inter / (a[2]*a[3] + b[2]*b[3] - inter + 1e-6)

# ================= PIPELINE =================
def detect_faces(frame, model):
    h, w = frame.shape[:2]
    blob = cv2.resize(frame, (300,300)).transpose(2,0,1)[None].astype(np.float32)
    out = model({model.inputs[0].any_name: blob})[model.outputs[0].any_name]
    faces = []
    for d in out[0][0]:
        if d[2] > 0.5:
            faces.append((int(d[3]*w), int(d[4]*h),
                          int(d[5]*w), int(d[6]*h)))
    return faces

def extract_embedding(crop, model):
    blob = cv2.resize(crop, (128,128)).transpose(2,0,1)[None].astype(np.float32)
    return model({model.inputs[0].any_name: blob})[
        model.outputs[0].any_name].flatten()

def aggregate_embeddings(embs):
    sims = np.array([[cosine(a,b) for b in embs] for a in embs])
    best = np.argmax(sims.sum(axis=1))
    good = [embs[i] for i in range(len(embs)) if sims[best][i] > 0.9]
    emb = np.mean(good, axis=0)
    return emb / np.linalg.norm(emb)

# ================= GALLERY MATCHING (NEW) =================
def best_gallery_match(emb, store, threshold):
    """
    Compare emb against every identity's gallery (list of embeddings)
    and return (gid, score) for the best match above threshold, else
    (None, 0). This replaces comparing against a single stored vector.
    """
    best_gid, best_score = None, 0.0
    for gid, gallery in store.items():
        score = max(cosine(emb, g) for g in gallery)
        if score > threshold and score > best_score:
            best_gid, best_score = gid, score
    return best_gid, best_score

def update_gallery(store, gid, emb):
    """
    Add emb to gid's gallery if it's sufficiently different from what's
    already stored (avoids saving near-duplicate frames), and cap the
    gallery at GALLERY_SIZE by dropping the oldest entry.
    """
    gallery = store[gid]
    if max(cosine(emb, g) for g in gallery) < GALLERY_ADD_THRESHOLD:
        gallery.append(emb)
        if len(gallery) > GALLERY_SIZE:
            gallery.pop(0)

# ================= CAMERA PROCESS =================
def process_camera(frame, tracks, potentials, models, is_entry, stable_frames):
    global next_gid

    active = set()
    faces = detect_faces(frame, models["face"])

    for (x1,y1,x2,y2) in faces:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        emb = extract_embedding(crop, models["reid"])
        box = (x1,y1,x2-x1,y2-y1)

        matched = False
        for tid, t in tracks.items():
            if iou(box, t["bbox"]) > IOU_THRESHOLD:
                t["bbox"] = box
                t["last_seen"] = time.time()
                active.add(tid)
                matched = True
                break

        if matched:
            continue

        if is_entry:
            pid = None
            for k, p in potentials.items():
                if iou(box, p["bbox"]) > IOU_THRESHOLD:
                    pid = k
                    break

            if pid is None:
                pid = len(potentials)
                potentials[pid] = {"bbox": box, "embs": [emb], "frames": 1}
                continue

            p = potentials[pid]
            p["frames"] += 1
            p["embs"].append(emb)
            p["bbox"] = box

            if p["frames"] >= stable_frames:
                final_emb = aggregate_embeddings(p["embs"])
                del potentials[pid]

                # NEW: match against each identity's gallery, not one vector
                gid, _ = best_gallery_match(final_emb, persistent_store, ENTRY_THRESHOLD)

                if gid is not None:
                    update_gallery(persistent_store, gid, final_emb)
                    save_store(persistent_store)
                    tracks[gid] = {"bbox": box, "last_seen": time.time()}
                    active.add(gid)
                else:
                    gid = next_gid
                    next_gid += 1
                    persistent_store[gid] = [final_emb]  # gallery starts with 1 entry
                    save_store(persistent_store)
                    tracks[gid] = {"bbox": box, "last_seen": time.time()}
                    active.add(gid)

        else:
            # NEW: match against gallery instead of single stored vector
            gid, _ = best_gallery_match(emb, persistent_store, SECONDARY_THRESHOLD)
            if gid is not None:
                update_gallery(persistent_store, gid, emb)
                tracks[gid] = {"bbox": box, "last_seen": time.time()}
                active.add(gid)

    # CHANGED: expire tracks based on elapsed time since last_seen,
    # not a frame-count "miss" counter. This behaves consistently
    # regardless of each camera's FPS.
    now = time.time()
    for tid in list(tracks.keys()):
        if tid not in active:
            if now - tracks[tid]["last_seen"] > MAX_MISSING_SECONDS:
                del tracks[tid]

# ================= CAMERA WORKER (NEW) =================
class CameraWorker:
    """
    NEW: wraps a single camera's capture handle, its own tracks dict,
    and (if it's an entry camera) its own potentials dict.
    Lets main() loop over an arbitrary list of cameras instead of
    hardcoding two separate blocks of near-duplicate code.
    """
    def __init__(self, config, stable_frames):
        self.name = config["name"]
        self.is_entry = config["is_entry"]
        self.cap = cv2.VideoCapture(config["source"])
        self.tracks = {}
        self.potentials = {} if self.is_entry else None
        self.stable_frames = stable_frames

    def step(self, models):
        ret, frame = self.cap.read()
        if not ret:
            return None
        process_camera(
            frame, self.tracks,
            self.potentials if self.is_entry else {},
            models, self.is_entry, self.stable_frames
        )
        for gid, t in self.tracks.items():
            x, y, w, h = t["bbox"]
            color = (0, 255, 0) if self.is_entry else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID {gid}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def release(self):
        self.cap.release()

# ================= MAIN =================
def main():
    models = load_models()

    # NEW: build a worker per camera in CAMERAS instead of two fixed captures
    probe_fps_cap = cv2.VideoCapture(CAMERAS[0]["source"])
    fps = probe_fps_cap.get(cv2.CAP_PROP_FPS)
    probe_fps_cap.release()
    stable_frames = int((fps if fps > 0 else 10) * STABLE_SECONDS)

    workers = [CameraWorker(cfg, stable_frames) for cfg in CAMERAS]

    while True:
        for w in workers:
            frame = w.step(models)
            if frame is not None:
                cv2.imshow(w.name, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    for w in workers:
        w.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()