# EXPLAIN.md

This document explains the full pipeline step-by-step, including how **manual CVAT annotations** are used to train the model.

## What the project does

Given a raw skateboarding video, the system:

- Runs **pose estimation** (YOLOv8-Pose)
- Computes simple **pose-derived motion/shape features** (e.g. `tuck_ratio`)
- Trains a classifier using **human-labeled trick intervals** from CVAT
- Runs inference to find trick moments and cuts highlight clips with FFmpeg

The goal is **core trick detection** (the “event”), and the highlight clip is produced by adding padding (e.g. ±2s) around detected events.

## The pipeline (scripts)

### 1) `src/01_ingest.py` — Upload the raw video (W&B Artifact)

What it does:

- Starts a W&B run (`job_type="ingest"`)
- Creates an artifact named `skate_session_raw` (`type="raw-footage"`)
- Adds your `.mp4` file to the artifact and uploads it

Why it matters:

- Makes the raw input reproducible and easy to fetch in later stages.

### 2) `src/02_process.py` — Extract pose metrics (W&B Table)

What it does:

- Downloads the latest `skate_session_raw` artifact
- Opens the video with OpenCV and runs YOLOv8-Pose on sampled frames
- Selects the “main” person (if multiple people are present) using:
  - **box_confidence × box_area**
- Applies guardrails (edge truncation, impossible ratios, etc.)
- Logs a W&B Table named `movement_metrics`

Key outputs:

- **Required columns** (kept stable so labels align):
  - `frame`: original video frame index (critical for alignment with CVAT)
  - `tuck_ratio`: \( \frac{leg\_len}{torso\_len} \) computed from normalized keypoints
  - `hip_y`: normalized hip y-position (when available)
- **Additional signals** (helpful for training and debugging):
  - Pose geometry: `torso_len`, `leg_len`, `shoulder_y`, `ankle_y`
  - Pose quality: `min_core_conf`, `mean_kpt_conf`, per-joint confidences
  - Box stats: `box_area`, `box_conf`, `box_cx`, `box_cy`
  - Filtering counters logged as `filter/*`

Sampling rate:

- Uses `sample_stride` (default `2`) which yields ~15fps effective sampling on a 30fps video.

Why this script is important for ML quality:

- If you sample too sparsely (e.g. ~6fps), you can miss fast trick dynamics.
- If you filter too aggressively, you can remove too many positive frames.

### 3) Manual labels (CVAT) — Create trick intervals

This is the supervision signal used for training.

#### What you label

- You label the **core trick window** only.
- You do *not* label the full highlight clip.
- Padding (e.g. ±2s) is added later when cutting clips.

#### How to label in CVAT (recommended approach)

1. Create a label named `trick`.
2. For each trick, create a **track** labeled `trick` spanning the trick interval:
   - Start at the first “trick begins” frame.
   - End at the last “landing/roll-away stabilizes” frame.
3. Export:
   - Export annotations as **CVAT for video 1.1**
4. Save as:
   - `raw_data/annotations.xml`

#### What the export looks like

CVAT exports tracks like:

- `<track label="trick">`
  - `<box frame="247" outside="0" ... />` (inside the interval)
  - ...
  - `<box frame="273" outside="1" ... />` (interval ended)

The important part is the **frame numbers**. The box coordinates do not matter for this project; the track is used purely as a time interval carrier.

### 4) `src/03_train.py` — Train using CVAT intervals (Model Artifact)

What it does:

1. Loads `movement_metrics` from a chosen W&B run (`--metrics-run-name`).
2. Parses `raw_data/annotations.xml` to get trick intervals in **frame space**.
3. Labels each row in the metrics table:
   - `is_trick = 1` if `frame` is inside any CVAT interval
   - `is_trick = 0` otherwise
4. Builds features:
   - `basic`: `tuck_ratio` only
   - `temporal`: rolling/delta features from `tuck_ratio` (+ `hip_y` when available)
   - `rich`: adds quality/body features from `02_process.py`
   - `rich_temporal`: rich + temporal
5. Uses a time-aware split (no random frame leakage):
   - trains on early frames, tests on the last chunk
6. Logs:
   - frame-level metrics (accuracy/precision/recall/AP)
   - event-level metrics (precision/recall by overlapping predicted events with GT intervals)
7. Saves and registers a model artifact:
   - `basic` saves a raw sklearn model
   - non-`basic` saves a payload containing:
     - `model`, `feature_columns`, `threshold`, etc.

Why interval labels work well here:

- You care about **events**, not single-frame classification.
- This enables event-level evaluation and clip cutting.

### 5) `src/04_action.py` — Inference + clip cutting (Highlights)

What it does:

1. Downloads:
   - model artifact `skate_trick_classifier:latest`
   - video artifact `skate_session_raw:latest`
2. Runs pose inference on sampled frames (`sample_stride`, default `2`).
3. Computes the same feature columns expected by the trained model:
   - supports payload models with `feature_columns`
4. Generates “trick moments”:
   - if `trick_prob > threshold`, count it as a candidate
   - confirms timestamps based on:
     - `required_streak`
     - `timestamp_cooldown_s`
5. Merges timestamps into clips:
   - merges if timestamps are within `merge_gap_s`
6. Cuts clips with FFmpeg:
   - adds padding (`clip_padding_s`) before/after each clip
7. Uploads highlight clips to W&B as an artifact

## How manual annotations map to training rows (the key alignment idea)

The “join key” between labels and features is:

- CVAT exports intervals in **video frame indices**
- `02_process.py` stores the original `frame_idx` as `frame`

So training labels are applied by:

- `is_trick = 1` iff `frame` ∈ any CVAT interval

This works even when you sample (e.g. every 2nd frame) because sampled frames still keep their original indices.

## Reproducibility / experimentation (W&B config knobs)

Both processing and action scripts publish knobs to `wandb.config`, making it easy to compare runs in the dashboard.

- `src/02_process.py`:
  - `sample_stride`, `edge_margin`, `min_torso_len`, `min_tuck_ratio`
- `src/04_action.py`:
  - `sample_stride`, `required_streak`, `default_threshold`, `timestamp_cooldown_s`
  - `merge_gap_s`, `clip_padding_s`

## Repository reference assets

The key files are:
- `raw_data/skate_video.mp4`: reference input video
- `raw_data/annotations.xml`: CVAT-exported trick intervals for the reference video

These provide a concrete example that others can run end-to-end.


