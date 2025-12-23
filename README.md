# 🛹 Skate-Clip Auto-Cropper

An automated MLOps pipeline that detects skateboard tricks in raw video footage and creates highlight clips using Computer Vision.

Built with **Weights & Biases**, **YOLOv8**, and **OpenCV**.

## 🚀 Overview

This project solves the problem of "sifting through hours of footage" to find tricks. It uses a 4-stage pipeline:
1.  **Ingest:** Uploads raw video assets to W&B Artifacts.
2.  **Process:** Extracts skeletal keypoints (Pose Estimation) and calculates "Tuck Ratios" to handle perspective distortion.
3.  **Train:** Trains a Random Forest classifier to distinguish "Tricks" from "Riding" based on physics metrics.
4.  **Action:** Applies the model to new footage, merges detections, and physically crops clips using FFmpeg.

## 🛠️ Architecture

| Stage | Script | Description | W&B Feature |
| :--- | :--- | :--- | :--- |
| **1. Ingest** | `src/01_ingest.py` | Uploads raw `.mp4` to cloud storage. | **Artifacts** |
| **2. Process** | `src/02_process.py` | Runs YOLOv8-Pose, cleans data (edge/ghost checks), and logs metrics. | **Tables** |
| **3. Train** | `src/03_train.py` | Trains Random Forest on clean data and versions the model. | **Model Registry** |
| **4. Action** | `src/04_action.py` | Downloads prod model, detects tricks, and cuts video clips. | **Automations** |

## 📦 Installation

### Prerequisites
* Python 3.10+
* FFmpeg (Required for video cutting)
    * `brew install ffmpeg` (macOS)
    * `sudo apt install ffmpeg` (Linux)

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/vazgabriel-wandb/skate-cropper.git](https://github.com/vazgabriel-wandb/skate-cropper.git)
    cd skate-cropper
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Configure Environment:
    Create a `.env` file in the root directory:
    ```env
    WANDB_API_KEY=your_api_key_here
    WANDB_PROJECT=your_wandb_project
    WANDB_ENTITY=your_organization
    ```

## 🏃 Usage

**1. Upload Raw Footage**
```bash
python3 src/01_ingest.py path/to/your/video.mp4
```

**2. Extract & Clean Data Runs Pose Estimation and filters out "garbage" data (occlusions, edge cases).**

```bash
python src/02_process.py
```

**3. Train the Model Retrains the classifier on the latest dataset and pushes a new version to the Registry.**

Note: Update the METRICS_RUN_NAME with the result from 02_process.py

```bash
python src/03_train.py
```

**4. Generate Highlights Downloads the latest production model and cuts the video.**

```bash
python src/04_action.py
```

Output clips will be saved in the `highlights/` folder.

## 🧠 Key Logic

* Tuck Ratio: Calculates Leg Length / Torso Length to detect jumps regardless of camera distance.
* Perspective Filters: Rejects detections where the skater touches the screen edges to prevent false positives.
* Merge Window: Automatically merges tricks that happen within 6 seconds of each other into a single continuous line.

## 📄 License

MIT