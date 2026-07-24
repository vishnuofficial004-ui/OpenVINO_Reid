import cv2
import numpy as np
import pickle
import os
import time
import json
from openvino.runtime import Core

# ================= CONFIG =================
STABLE_SECONDS = 1.5       # how long a face must stay put before we try to identify it
IOU_THRESHOLD = 0.3
ENTRY_THRESHOLD = 0.80
SECONDARY_THRESHOLD = 0.72
MAX_MISSING_SECONDS = 3.0      # drop a track if unseen for this long
REENTRY_WINDOW_SECONDS = 10.0  # how long a lost identity counts as "still nearby"
MATCH_MARGIN = 0.05            # min score gap needed to accept a match over the runner-up

EMBEDDING_FILE = "embeddings_store.pkl"
GALLERY_SIZE = 5               # max embeddings kept per identity
GALLERY_ADD_THRESHOLD = 0.85   # only add to gallery if sufficiently different from existing entries

CAMERAS = [
    {"name": "MOBILE_ENTRY", "source": "http://192.168.1.5:8080/video", "is_entry": True},   # entry cams register new identities
    {"name": "PC_SECONDARY",  "source": 0,                               "is_entry": False},  # secondary cams only match existing ones
    {"name": "HALLWAY_CAM",   "source": "rtsp://192.168.1.20/stream",    "is_entry": False},  # third camera, confirms N-camera scale
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

event_log = []  # record of register/match/lost events, used for the continuity metric

def log_event(event_type, gid, camera_name, extra=None):
    event_log.append({
        "timestamp": time.time(),
        "event": event_type,
        "gid": gid,
        "camera": camera_name,
        "extra": extra or {},
    })

def compute_continuity(log):
    # % of losses that were later reconnected via a re-entry match
    lost_events = [e for e in log if e["event"] == "lost"]
    matched_events = [e for e in log if e["event"] == "matched"]

    reconnected = 0
    for lost in lost_events:
        found = any(
            m["gid"] == lost["gid"]
            and m["timestamp"] > lost["timestamp"]
            and m["extra"].get("reentry") is True
            for m in matched_events
        )
        if found:
            reconnected += 1

    total = len(lost_events)
    pct = (reconnected / total * 100) if total > 0 else 0.0

    return {
        "total_losses": total,
        "reconnected": reconnected,
        "continuity_pct": round(pct, 2),
    }

def load_ground_truth(path):
    # validates schema only, replay/comparison happens elsewhere
    with open(path, "r") as f:
        data = json.load(f)

    if "clips" not in data or "crossings" not in data:
        raise ValueError("ground truth file must have 'clips' and 'crossings' keys")

    clip_ids = set()
    for clip in data["clips"]:
        for key in ("clip_id", "camera", "video_path", "identities"):
            if key not in clip:
                raise ValueError(f"clip missing required field: {key}")
        clip_ids.add(clip["clip_id"])

    for crossing in data["crossings"]:
        for key in ("person_label", "from_clip", "to_clip", "expected_same_identity"):
            if key not in crossing:
                raise ValueError(f"crossing missing required field: {key}")
        if crossing["from_clip"] not in clip_ids or crossing["to_clip"] not in clip_ids:
            raise ValueError(
                f"crossing references unknown clip_id: "
                f"{crossing['from_clip']} or {crossing['to_clip']}"
            )

    return data

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

def best_gallery_match(emb, store, threshold):
    # returns (gid, score); refuses to match if top two candidates are too close
    scores = []
    for gid, gallery in store.items():
        score = max(cosine(emb, g) for g in gallery)
        if score > threshold:
            scores.append((gid, score))

    if not scores:
        return None, 0.0

    scores.sort(key=lambda x: x[1], reverse=True)
    best_gid, best_score = scores[0]

    if len(scores) > 1:
        second_score = scores[1][1]
        if (best_score - second_score) < MATCH_MARGIN:
            return None, 0.0

    return best_gid, best_score

def update_gallery(store, gid, emb):
    gallery = store[gid]
    if max(cosine(emb, g) for g in gallery) < GALLERY_ADD_THRESHOLD:
        gallery.append(emb)
        if len(gallery) > GALLERY_SIZE:
            gallery.pop(0)

def check_reentry(recently_lost, gid):
    # pops gid so it can't be reused for a second match
    lost_at = recently_lost.pop(gid, None)
    if lost_at is None:
        return False
    return (time.time() - lost_at) <= REENTRY_WINDOW_SECONDS

# ================= CAMERA PROCESS =================
def process_camera(frame, tracks, potentials, models, is_entry, stable_frames, recently_lost, camera_name):
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

                gid, _ = best_gallery_match(final_emb, persistent_store, ENTRY_THRESHOLD)

                if gid is not None:
                    update_gallery(persistent_store, gid, final_emb)
                    save_store(persistent_store)
                    reentry = check_reentry(recently_lost, gid)
                    tracks[gid] = {"bbox": box, "last_seen": time.time(), "reentry": reentry}
                    active.add(gid)
                    log_event("matched", gid, camera_name, {"reentry": reentry})
                else:
                    gid = next_gid
                    next_gid += 1
                    persistent_store[gid] = [final_emb]
                    save_store(persistent_store)
                    tracks[gid] = {"bbox": box, "last_seen": time.time(), "reentry": False}
                    active.add(gid)
                    log_event("registered", gid, camera_name)

        else:
            gid, _ = best_gallery_match(emb, persistent_store, SECONDARY_THRESHOLD)
            if gid is not None:
                update_gallery(persistent_store, gid, emb)
                reentry = check_reentry(recently_lost, gid)
                tracks[gid] = {"bbox": box, "last_seen": time.time(), "reentry": reentry}
                active.add(gid)
                log_event("matched", gid, camera_name, {"reentry": reentry})

    now = time.time()
    for tid in list(tracks.keys()):
        if tid not in active:
            if now - tracks[tid]["last_seen"] > MAX_MISSING_SECONDS:
                recently_lost[tid] = now
                log_event("lost", tid, camera_name)
                del tracks[tid]

    for tid in list(recently_lost.keys()):
        if now - recently_lost[tid] > REENTRY_WINDOW_SECONDS:
            del recently_lost[tid]

# ================= CAMERA WORKER =================
class CameraWorker:
    def __init__(self, config, stable_frames):
        self.name = config["name"]
        self.is_entry = config["is_entry"]
        self.cap = cv2.VideoCapture(config["source"])
        self.tracks = {}
        self.potentials = {} if self.is_entry else None
        self.stable_frames = stable_frames

    def step(self, models, recently_lost):
        ret, frame = self.cap.read()
        if not ret:
            return None
        process_camera(
            frame, self.tracks,
            self.potentials if self.is_entry else {},
            models, self.is_entry, self.stable_frames,
            recently_lost, self.name
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

    probe_fps_cap = cv2.VideoCapture(CAMERAS[0]["source"])
    fps = probe_fps_cap.get(cv2.CAP_PROP_FPS)
    probe_fps_cap.release()
    stable_frames = int((fps if fps > 0 else 10) * STABLE_SECONDS)

    workers = [CameraWorker(cfg, stable_frames) for cfg in CAMERAS]
    shared_recently_lost = {}  # shared across cameras so cross-camera re-entry is detected

    while True:
        for w in workers:
            frame = w.step(models, shared_recently_lost)
            if frame is not None:
                cv2.imshow(w.name, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    for w in workers:
        w.release()
    cv2.destroyAllWindows()

    stats = compute_continuity(event_log)
    print(f"Identity continuity: {stats['reconnected']}/{stats['total_losses']} "
          f"reconnected ({stats['continuity_pct']}%)")

if __name__ == "__main__":
    main()