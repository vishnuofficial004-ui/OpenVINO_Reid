# Multi-Camera Person Tracking System (OpenVINO, CPU)

## 1. Overview

This project implements a **real-time multi-camera person tracking system** using **OpenVINO on CPU**.  
It assigns **stable global IDs** to people detected in a **primary (entry) camera** and continues tracking the same individuals in **secondary cameras**, using **face re-identification** with **body detection fallback**.

The system is designed for:
- Stable global ID assignment
- CPU-only deployment
- Minimal FPS drop
- No unnecessary frame resizing
- Clear, human-readable visual overlays

---

## 2. Key Features

- Face detection–based tracking
- Global ID assignment in primary camera
- Cross-camera re-identification
- Body detection fallback when face tracking fails
- No ID reassignment while a person is actively tracked
- Thin color-coded bounding boxes:
  - **Red** → Primary camera
  - **Green** → Secondary camera
- Confidence score displayed above each bounding box
- Synchronized timestamp overlay (bottom-right corner)
- Optimized for real-time CPU performance

---

## 3. System Requirements

### Hardware
- Intel CPU (recommended)
- One or more USB / IP cameras

### Software
- Python 3.8 – 3.11
- OpenVINO Runtime
- OpenCV
- NumPy
- SciPy

---

## 4. Model Downloads (Direct Links)

Download the following **OpenVINO IR models** from Intel Open Model Zoo.

### Face Detection
**Model:** `face-detection-retail-0004`  
Download:
https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-retail-0004/FP16/


### Face Re-Identification
**Model:** `face-reidentification-retail-0095`  
Download:
https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-reidentification-retail-0095/FP16/


### Person (Body) Detection
**Model:** `person-detection-retail-0013`  
Download:
https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-retail-0013/FP16/


---

## 5. Model Directory Structure

After downloading, organize models as follows:

<pre>
models/
├── face-detection-retail-0004/
│ ├── face-detection-retail-0004.xml
│ └── face-detection-retail-0004.bin
├── face-reidentification-retail-0095/
│ ├── face-reidentification-retail-0095.xml
│ └── face-reidentification-retail-0095.bin
└── person-detection-retail-0013/
├── person-detection-retail-0013.xml
└── person-detection-retail-0013.bin
</pre>


---

## 6. Installation

### Install requirements
pip install -r requirements.txt

### 7. Camera Configuration

Edit camera sources in the code:

ENTRY_CAMERAS = {
    "ENTRY_1": 0
}

SECONDARY_CAMERAS = {
    "SEC_1": 1
}

8. Running the Application
python main.py

9. Visual Output
  Bounding Boxes
  Thin rectangles
  Red for primary camera
  Green for secondary camera

Label Above Bounding Box
ID <global_id> | <confidence>

Timestamp (Bottom Right)
HH:MM:SS
YYYY-MM-DD


Timestamp is synchronized across all cameras.

10. ID Assignment Logic

   Global IDs are created only in the primary camera
   Secondary cameras only match existing IDs
   If a person is already tracked:
   Re-identification is skipped
   Global ID is preserved
   Face tracking has priority
   Body detection is used only when face tracking fails
   IDs are never reassigned or recycled during runtime
