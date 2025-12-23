# 🛹 Skate-Clip Auto-Cropper

An automated MLOps pipeline that detects skateboard tricks in raw video footage and creates highlight clips using Computer Vision.

Built with **Weights & Biases**, **YOLOv8**, and **OpenCV**.

## 🚀 Overview

This project solves the problem of "sifting through hours of footage" to find tricks. It uses a 4-stage pipeline:
1.  **Ingest:** Uploads raw video assets to W&B Artifacts.
2.  **Process:** Runs YOLOv8-Pose, computes pose-derived features (including `tuck_ratio`), and logs a metrics table to W&B.
3.  **Train:** Trains a classifier using **manual CVAT annotations** (trick time intervals) aligned to the metrics table.
4.  **Action:** Applies the model to new footage, merges detections, and crops highlight clips using FFmpeg.

## 🛠️ Architecture

| Stage | Script | Description | W&B Feature |
| :--- | :--- | :--- | :--- |
| **1. Ingest** | `src/01_ingest.py` | Uploads raw `.mp4` to cloud storage. | **Artifacts** |
| **2. Process** | `src/02_process.py` | Runs YOLOv8-Pose, cleans data, logs pose metrics + quality signals. | **Tables** |
| **3. Train** | `src/03_train.py` | Trains a supervised classifier using CVAT-labeled trick intervals. | **Model Registry** |
| **4. Action** | `src/04_action.py` | Loads the model, runs inference, merges events, cuts clips. | **Automations** |

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

**2. Extract & Clean Data**

Runs pose estimation, filters out bad detections, and logs a W&B Table called `movement_metrics`.

Defaults are tuned for ~30fps footage and run at ~15fps effective sampling (`sample_stride=2`).

```bash
python src/02_process.py
```

**3. Label tricks (manual CVAT annotations)**

This project uses **event-level labels** (core trick windows) created in CVAT and exported to XML.

- Annotate each trick as a **track** labeled `trick` across the core trick interval (frame range).
- Export as **CVAT for video 1.1**.
- Ensure the export contains `<track ... label="trick">` entries.
- Save the file locally as `raw_data/annotations.xml`.

Reference files (optional, for this repo’s example clip):
- `raw_data/skate_video.mp4`
- `raw_data/annotations.xml`

**4. Train the model (supervised)**

Note: Update `--metrics-run-name` with the run name produced by `02_process.py` (shown in W&B UI).

```bash
python src/03_train.py --metrics-run-name <YOUR_METRICS_RUN_NAME> --annotations-xml raw_data/annotations.xml
```

Recommended settings (works with the current inference code):
- `--feature-set temporal` (adds rolling/delta features for better fast-trick detection)
- Optional: `--min-core-conf 0.2` to drop low-confidence pose rows
- Optional: `--dropna` when using rich features

Available feature sets:
- `basic`: `tuck_ratio` only (legacy/simple)
- `temporal`: temporal features from `tuck_ratio` (+ `hip_y` when available)
- `rich`: adds pose quality/body signals from `02_process.py` (e.g. confidences, bbox stats)
- `rich_temporal`: `rich` + temporal features

**5. Generate highlights**

```bash
python src/04_action.py
```

Output clips will be saved in the `highlights/` folder.

### W&B config knobs (compare runs easily)
Both `src/02_process.py` and `src/04_action.py` write key knobs into `wandb.config` so you can compare runs in the dashboard.

- `src/02_process.py`:
  - `sample_stride`, `edge_margin`, `min_torso_len`, `min_tuck_ratio`
- `src/04_action.py`:
  - `sample_stride`, `required_streak`, `default_threshold`, `timestamp_cooldown_s`
  - `merge_gap_s`, `clip_padding_s`

## 🧠 Key Logic

* Tuck Ratio: Calculates Leg Length / Torso Length to detect jumps regardless of camera distance.
* Perspective Filters: Rejects detections where the skater touches the screen edges to prevent false positives.
* Merge Window: Merges detected trick moments into clips (default `merge_gap_s=6.0`).

For a deeper walkthrough, see [EXPLAIN.md](./EXPLAIN.md).

## 📄 License

MIT