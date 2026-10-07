# Multi-Camera Face Re-Identification

A small computer vision pipeline that tries to keep a person's identity consistent as they move between two (or more) camera feeds. The idea is simple: see a face on the entry camera, register it, and if the same face shows up later on a different camera, recognize it as the same person instead of treating it as someone new.

Built on OpenVINO for face detection and re-identification, with a lightweight in-memory + pickle-based store for identities. No backend, no database, no deployment story — this is a local pipeline you run with two camera sources pointed at your face.

## Why

Most face re-ID demos online do single-camera tracking (one feed, boxes with IDs that persist frame to frame). The harder and more interesting problem is **cross-camera handoff**: the same person walking out of one camera's view and into another's, possibly minutes later, and the system still knowing it's the same person. That's what this project is actually about.

## How it works

**1. Detection + embedding.** Every frame goes through a face detector (`face-detection-retail-0004`). Each detected face is cropped and passed through a re-identification model (`face-reidentification-retail-0095`) that outputs a 256-d embedding vector.

**2. Entry vs. secondary cameras.** Cameras are split into two roles:
- **Entry camera** — where new identities get registered. A face has to stay in frame for `STABLE_SECONDS` (default 1.5s) before the pipeline commits to identifying it, to avoid registering junk off a single noisy frame.
- **Secondary camera(s)** — only try to match against people who already exist in the store. They never register anyone new.

**3. Gallery-based matching, not a single average vector.** Early versions of this stored one averaged embedding per person. That's fragile — lighting, angle, and expression all shift the embedding, and averaging blurs out the real variation. Instead, each identity keeps a small **gallery** of up to `GALLERY_SIZE` (5) distinct embeddings. A new face is compared against every embedding in every identity's gallery, and the best score wins. The gallery only accepts a new embedding if it's sufficiently different from what's already stored (`GALLERY_ADD_THRESHOLD`), so it doesn't fill up with five near-identical frames from the same two seconds of video.

**4. Ambiguity guard.** If the top match and the second-best match are too close to each other (within `MATCH_MARGIN`), the pipeline refuses to pick one. It's better to not match than to silently merge two different people because their embeddings happened to be close.

**5. Time-based track expiry, not frame counts.** A track is dropped after `MAX_MISSING_SECONDS` of not being seen, rather than after N missed frames. Frame-count thresholds break the moment camera FPS changes; time doesn't.

**6. Re-entry detection across cameras.** When someone's track expires, they go into a shared `recently_lost` pool (shared across *all* cameras, not per-camera) for `REENTRY_WINDOW_SECONDS`. If they're matched again — on any camera — within that window, it's logged as a re-entry rather than a fresh appearance. This is the actual "did we keep the identity continuous across a handoff" signal.

**7. Event log + continuity metric.** Every register / match / loss event is timestamped and logged. At the end of a run, `compute_continuity()` reports what percentage of "lost" events were later reconnected via a re-entry match. This is meant as a real, computable metric against labeled footage — not a number I made up for a resume.

## Project structure

```
pipeline.py                 the pipeline — detection, matching, tracking, everything
ground_truth_schema.md      schema for labeling real test clips to validate against
ground_truth_example.json   example (placeholder — not real recorded data)
```

## Setup

```bash
pip install opencv-python numpy openvino
```

Download the OpenVINO models and place them under `models/`:
- `face-detection-retail-0004`
- `face-reidentification-retail-0095`

(Both are available from the [OpenVINO Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo).)

Edit the `CAMERAS` list in `pipeline.py` to point at your actual camera sources — an IP/RTSP/DroidCam-style URL for a phone, an integer index for a local webcam:

```python
CAMERAS = [
    {"name": "MOBILE_ENTRY", "source": "http://192.168.1.5:8080/video", "is_entry": True},
    {"name": "PC_SECONDARY",  "source": 0, "is_entry": False},
]
```

Run it:

```bash
python pipeline.py
```

Press `Esc` to quit. On exit, it prints the identity continuity stat computed from the session's event log.

## Known limitations (honestly)

This is a working prototype, not a production system. Things it does **not** do yet:

- **No concurrency.** The main loop reads and processes each camera sequentially, one frame at a time. With more than a couple of cameras, this becomes the bottleneck — a slower camera or a slow frame on one source delays every other camera's next read. A producer/consumer setup (each camera on its own thread feeding a queue, a separate thread doing the matching) would fix this but isn't built.
- **Not validated against real labeled footage yet.** The continuity metric is implemented and has been checked against synthetic event logs, but it hasn't been run against actual recorded multi-camera clips with ground-truth labels. `ground_truth_schema.md` defines the format for that; `ground_truth_example.json` is a placeholder, not real data.
- **One known gap in the ambiguity guard.** On the entry camera, if a match is rejected for being too ambiguous (margin check fails), the current code falls through to registering a brand-new identity. It should instead be treated as "uncertain, don't create a new identity either" — this hasn't been fixed yet.
- **No re-ranking or ANN index.** Matching is a brute-force scan over every identity's gallery (`O(N × G)` per face). Fine at small scale; would need something like FAISS or an exact flat index at a few hundred+ identities.
- **Single-process, local-only.** No API, no multi-user support, no persistence beyond a local pickle file.

## Why gallery-based matching over a single averaged embedding

The original version of this stored exactly one embedding per person — an average over the first few seconds they were seen. The problem: averaging embeddings from different lighting/angles doesn't give you a "typical" face, it gives you a blurred-out point that may not be close to *any* of the real embeddings, including the person's own future appearances.

The gallery approach instead keeps a handful of distinct real embeddings per person and matches against the best one. It costs more compute per match (`O(N × G)` instead of `O(N)`), but at the identity-store sizes this is meant for (tens, maybe low hundreds of people), that's negligible next to the cost of running the detection and re-ID models themselves.

## License

MIT (This is a learning project).
