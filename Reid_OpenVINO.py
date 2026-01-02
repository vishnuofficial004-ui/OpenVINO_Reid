import cv2
import numpy as np
import pickle
import os
from openvino.runtime import Core

# ================= CONFIG =================
STABLE_SECONDS = 1.5
IOU_THRESHOLD = 0.3
ENTRY_THRESHOLD = 0.80
SECONDARY_THRESHOLD = 0.72
MAX_MISSING = 5

MOBILE_STREAM = "http://192.168.1.5:8080/video"  # CHANGE IP
PC_CAM_INDEX = 0
EMBEDDING_FILE = "embeddings_store.pkl"

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

        # Already tracked?
        matched = False
        for tid, t in tracks.items():
            if iou(box, t["bbox"]) > IOU_THRESHOLD:
                t["bbox"] = box
                t["miss"] = 0
                active.add(tid)
                matched = True
                break

        if matched:
            continue

        # ENTRY camera
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

                for gid, e in persistent_store.items():
                    if cosine(final_emb, e) > ENTRY_THRESHOLD:
                        tracks[gid] = {"bbox": box, "miss": 0}
                        active.add(gid)
                        break
                else:
                    gid = next_gid
                    next_gid += 1
                    persistent_store[gid] = final_emb
                    save_store(persistent_store)
                    tracks[gid] = {"bbox": box, "miss": 0}
                    active.add(gid)

        # SECONDARY camera
        else:
            for gid, e in persistent_store.items():
                if cosine(emb, e) > SECONDARY_THRESHOLD:
                    tracks[gid] = {"bbox": box, "miss": 0}
                    active.add(gid)
                    break

    for tid in list(tracks.keys()):
        if tid not in active:
            tracks[tid]["miss"] += 1
            if tracks[tid]["miss"] > MAX_MISSING:
                del tracks[tid]

# ================= MAIN =================
def main():
    models = load_models()
    cap_m = cv2.VideoCapture(MOBILE_STREAM)
    cap_p = cv2.VideoCapture(PC_CAM_INDEX)

    tracks_m, tracks_p = {}, {}
    potentials = {}

    fps = cap_m.get(cv2.CAP_PROP_FPS)
    stable_frames = int((fps if fps > 0 else 10) * STABLE_SECONDS)

    while True:
        rm, fm = cap_m.read()
        rp, fp = cap_p.read()

        if rm:
            process_camera(fm, tracks_m, potentials, models, True, stable_frames)
            for gid, t in tracks_m.items():
                x,y,w,h = t["bbox"]
                cv2.rectangle(fm,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(fm,f"ID {gid}",(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            cv2.imshow("MOBILE (ENTRY)", fm)

        if rp:
            process_camera(fp, tracks_p, {}, models, False, stable_frames)
            for gid, t in tracks_p.items():
                x,y,w,h = t["bbox"]
                cv2.rectangle(fp,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(fp,f"ID {gid}",(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
            cv2.imshow("PC (SECONDARY)", fp)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap_m.release()
    cap_p.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

