# The Automation: Uses Registry Model -> Crops Video
import wandb
import cv2
import joblib
import pandas as pd
import subprocess
import os
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()


def create_highlights():
    # 1. Initialize the "Automation" Job
    run = wandb.init(project="skate-cropper", job_type="automation")

    # ----------------------------
    # Config knobs (W&B)
    # ----------------------------
    # These show up in the W&B dashboard and let you compare runs easily.
    default_cfg = {
        # inference cadence
        "sample_stride": 2,  # ~15fps on 30fps video
        # detection behavior
        "required_streak": 1,  # "never miss" mode
        "default_threshold": 0.5,  # used if model artifact doesn't ship a threshold
        "timestamp_cooldown_s": 1.0,
        # clip generation
        "merge_gap_s": 6.0,  # merge confirmed timestamps into the same clip if within this gap
        "clip_padding_s": 2.0,  # +/- seconds around merged clip
        # debug
        "debug_window_start_s": 8.0,
        "debug_window_end_s": 12.0,
    }
    run.config.update(default_cfg, allow_val_change=True)

    # ------------------------------------------------------------------
    # STEP A: LOAD ASSETS (The "Brain" and the "Raw Material")
    # ------------------------------------------------------------------

    # Download Model from Registry
    # Replace 'skate_trick_classifier:latest' with your specific registry link if needed
    model_artifact = run.use_artifact("skate_trick_classifier:latest")
    model_dir = model_artifact.download()
    model = joblib.load(os.path.join(model_dir, "trick_classifier.pkl"))

    # Download Video
    video_artifact = run.use_artifact("skate_session_raw:latest")
    video_path = (
        video_artifact.download() + "/skate_video.mp4"
    )  # Check filename matches upload

    # ------------------------------------------------------------------
    # STEP B: THE INFERENCE LOOP (See -> Think)
    # ------------------------------------------------------------------
    print("Scanning video for tricks...")
    yolo = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    trick_timestamps = []
    frame_idx = 0

    consecutive_trick_frames = 0
    # Read knobs from config
    sample_stride = int(run.config.get("sample_stride", 2))
    REQUIRED_STREAK = int(run.config.get("required_streak", 1))
    DEFAULT_THRESHOLD = float(run.config.get("default_threshold", 0.5))
    TIMESTAMP_COOLDOWN_S = float(run.config.get("timestamp_cooldown_s", 1.0))
    MERGE_GAP_S = float(run.config.get("merge_gap_s", 6.0))
    CLIP_PADDING_S = float(run.config.get("clip_padding_s", 2.0))
    DEBUG_WINDOW_START_S = float(run.config.get("debug_window_start_s", 8.0))
    DEBUG_WINDOW_END_S = float(run.config.get("debug_window_end_s", 12.0))

    # COCO Keypoint Indices
    LEFT_HIP, RIGHT_HIP = 11, 12
    LEFT_ANKLE, RIGHT_ANKLE = 15, 16
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

    # Support both:
    # - bare sklearn model (legacy/basic)
    # - payload dict with metadata (rich/temporal)
    model_payload = None
    if isinstance(model, dict) and "model" in model:
        model_payload = model
        model = model_payload["model"]
        feature_columns = model_payload.get("feature_columns", ["tuck_ratio"])
        threshold = float(model_payload.get("threshold", DEFAULT_THRESHOLD))
    else:
        feature_columns = ["tuck_ratio"]
        threshold = DEFAULT_THRESHOLD

    # Log the effective decision threshold (payload may override default)
    run.log({"inference/threshold": threshold})

    # rolling state for temporal features
    from collections import deque

    ratio_hist = deque(maxlen=5)
    ratio_diff1_hist = deque(maxlen=5)
    hip_hist = deque(maxlen=5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_idx = frame_idx
        frame_idx += 1

        # Calculate current time
        current_time = current_frame_idx / fps

        # Debug window
        debug_mode = DEBUG_WINDOW_START_S < current_time < DEBUG_WINDOW_END_S

        # Analyze every Nth frame (Matches processing logic)
        if current_frame_idx % sample_stride == 0:
            results = yolo(frame, verbose=False)

            # Default assumption: No trick detected
            is_trick_frame = False

            if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
                # Pick main person (same heuristic as processing): max(conf * area)
                chosen_idx = 0
                try:
                    boxes = results[0].boxes
                    if (
                        boxes is not None
                        and boxes.xyxy is not None
                        and len(boxes.xyxy) > 0
                    ):
                        xyxy_all = boxes.xyxy.cpu().numpy()
                        conf_all = (
                            boxes.conf.cpu().numpy()
                            if boxes.conf is not None
                            else np.ones(len(xyxy_all))
                        )
                        areas = (xyxy_all[:, 2] - xyxy_all[:, 0]) * (
                            xyxy_all[:, 3] - xyxy_all[:, 1]
                        )
                        chosen_idx = int(np.argmax(conf_all * areas))
                except Exception:
                    chosen_idx = 0

                kpts = results[0].keypoints.xyn.cpu().numpy()[chosen_idx]
                kpts_conf = None
                if getattr(results[0].keypoints, "conf", None) is not None:
                    kpts_conf = results[0].keypoints.conf.cpu().numpy()[chosen_idx]

                # Sanity check (same as training)
                if (
                    kpts[LEFT_SHOULDER][0] > 0
                    and kpts[LEFT_HIP][0] > 0
                    and kpts[LEFT_ANKLE][0] > 0
                ):
                    shoulder_y = (kpts[LEFT_SHOULDER][1] + kpts[RIGHT_SHOULDER][1]) / 2
                    hip_y = (kpts[LEFT_HIP][1] + kpts[RIGHT_HIP][1]) / 2
                    ankle_y = (kpts[LEFT_ANKLE][1] + kpts[RIGHT_ANKLE][1]) / 2

                    # --- GUARDRAIL 2: EDGE CHECK ---
                    # Reject if body parts are too close to top/bottom edges
                    edge_margin = 0.02
                    in_safe_zone = (
                        shoulder_y > edge_margin
                        and shoulder_y < 1.0 - edge_margin
                        and hip_y > edge_margin
                        and hip_y < 1.0 - edge_margin
                        and ankle_y > edge_margin
                        and ankle_y < 1.0 - edge_margin
                    )

                    if not in_safe_zone:
                        if debug_mode:
                            print(f"[{current_time:.2f}s] REJECTED: Touching Edge")
                        continue

                    torso_len = abs(hip_y - shoulder_y)
                    leg_len = abs(ankle_y - hip_y)

                    if torso_len > 0.01 and leg_len > 0.01:
                        tuck_ratio = float(leg_len / torso_len)

                        # Reject impossible ratios (like 0.06)
                        if tuck_ratio <= 0.2:
                            if frame_idx % 60 == 0:
                                print(
                                    f"Frame {frame_idx}: Rejected (Impossible Ratio: {tuck_ratio:.2f})"
                                )
                            continue

                        # Prepare row for the Brain (Must match training features!)
                        # Build a single-row feature dict matching training.
                        feat = {
                            "tuck_ratio": tuck_ratio,
                        }
                        # quality/body signals (if model expects them)
                        feat["hip_y"] = float(hip_y)
                        feat["torso_len"] = float(torso_len)
                        feat["leg_len"] = float(leg_len)

                        if kpts_conf is not None:
                            left_shoulder_conf = float(kpts_conf[LEFT_SHOULDER])
                            right_shoulder_conf = float(kpts_conf[RIGHT_SHOULDER])
                            left_hip_conf = float(kpts_conf[LEFT_HIP])
                            right_hip_conf = float(kpts_conf[RIGHT_HIP])
                            left_ankle_conf = float(kpts_conf[LEFT_ANKLE])
                            right_ankle_conf = float(kpts_conf[RIGHT_ANKLE])
                            feat["left_shoulder_conf"] = left_shoulder_conf
                            feat["right_shoulder_conf"] = right_shoulder_conf
                            feat["left_hip_conf"] = left_hip_conf
                            feat["right_hip_conf"] = right_hip_conf
                            feat["left_ankle_conf"] = left_ankle_conf
                            feat["right_ankle_conf"] = right_ankle_conf
                            feat["mean_kpt_conf"] = float(np.nanmean(kpts_conf))
                            feat["min_core_conf"] = float(
                                np.nanmin(
                                    [
                                        left_shoulder_conf,
                                        right_shoulder_conf,
                                        left_hip_conf,
                                        right_hip_conf,
                                        left_ankle_conf,
                                        right_ankle_conf,
                                    ]
                                )
                            )
                        else:
                            feat["mean_kpt_conf"] = np.nan
                            feat["min_core_conf"] = np.nan

                        # box stats
                        feat["box_area"] = np.nan
                        feat["box_conf"] = np.nan
                        feat["box_cx"] = np.nan
                        feat["box_cy"] = np.nan
                        try:
                            boxes = results[0].boxes
                            if (
                                boxes is not None
                                and boxes.xyxy is not None
                                and len(boxes.xyxy) > chosen_idx
                            ):
                                x1, y1, x2, y2 = [
                                    float(v)
                                    for v in boxes.xyxy.cpu().numpy()[chosen_idx]
                                ]
                                feat["box_area"] = float(
                                    max(0.0, x2 - x1) * max(0.0, y2 - y1)
                                )
                                if (
                                    boxes.conf is not None
                                    and len(boxes.conf) > chosen_idx
                                ):
                                    feat["box_conf"] = float(
                                        boxes.conf.cpu().numpy()[chosen_idx]
                                    )
                                feat["box_cx"] = float((x1 + x2) / 2.0)
                                feat["box_cy"] = float((y1 + y2) / 2.0)
                        except Exception:
                            pass

                        # temporal features (computed from histories)
                        ratio_hist.append(tuck_ratio)
                        diff1 = tuck_ratio - (
                            ratio_hist[-2] if len(ratio_hist) >= 2 else tuck_ratio
                        )
                        ratio_diff1_hist.append(diff1)
                        feat["tuck_ratio_diff1"] = float(diff1)
                        diff2 = diff1 - (
                            ratio_diff1_hist[-2]
                            if len(ratio_diff1_hist) >= 2
                            else diff1
                        )
                        feat["tuck_ratio_diff2"] = float(diff2)

                        # rolling stats over up to last 5 samples
                        r_arr = np.array(ratio_hist, dtype=float)
                        feat["tuck_ratio_roll_mean3"] = (
                            float(np.mean(r_arr[-3:])) if len(r_arr) else 0.0
                        )
                        feat["tuck_ratio_roll_std3"] = (
                            float(np.std(r_arr[-3:], ddof=0)) if len(r_arr) else 0.0
                        )
                        feat["tuck_ratio_roll_mean5"] = (
                            float(np.mean(r_arr)) if len(r_arr) else 0.0
                        )
                        feat["tuck_ratio_roll_std5"] = (
                            float(np.std(r_arr, ddof=0)) if len(r_arr) else 0.0
                        )

                        hip_hist.append(float(hip_y))
                        h_arr = np.array(hip_hist, dtype=float)
                        feat["hip_y_diff1"] = (
                            float(h_arr[-1] - h_arr[-2]) if len(h_arr) >= 2 else 0.0
                        )
                        feat["hip_y_roll_mean3"] = (
                            float(np.mean(h_arr[-3:])) if len(h_arr) else 0.0
                        )
                        feat["hip_y_roll_std3"] = (
                            float(np.std(h_arr[-3:], ddof=0)) if len(h_arr) else 0.0
                        )

                        row = [feat.get(c, np.nan) for c in feature_columns]
                        df = pd.DataFrame([row], columns=feature_columns)

                        probs = model.predict_proba(df)[0]
                        trick_prob = probs[1]  # Probability of class 1 (Trick)

                        if debug_mode:
                            status = "NO"
                            if trick_prob > threshold:
                                status = "YES"
                            print(
                                f"[{current_time:.2f}s] Ratio: {tuck_ratio:.2f} | Torso: {torso_len:.3f} | Conf: {trick_prob:.2f} -> {status}"
                            )

                        # CHANGE 2: Check Threshold
                        if trick_prob > threshold:
                            is_trick_frame = True
                            print(
                                f"Potential detected at {current_frame_idx/fps:.2f}s (Conf: {trick_prob:.2f})"
                            )

            if is_trick_frame:
                consecutive_trick_frames += 1
            else:
                consecutive_trick_frames = 0  # Reset streak if we miss a frame

            # Only record timestamp if streak is met
            if consecutive_trick_frames >= REQUIRED_STREAK:
                # Avoid duplicate timestamps for the same trick
                if not trick_timestamps or (
                    current_time - trick_timestamps[-1] > TIMESTAMP_COOLDOWN_S
                ):
                    print(f"*** TRICK CONFIRMED at {current_time:.2f}s ***")
                    trick_timestamps.append(current_time)
    cap.release()

    # ------------------------------------------------------------------
    # STEP C: THE DIRECTOR (Logic to merge timestamps into clips)
    # ------------------------------------------------------------------
    if not trick_timestamps:
        print("No tricks detected :(")
        run.finish()
        return

    print(f"Detected {len(trick_timestamps)} trick frames. Merging into clips...")

    # Simple logic: If timestamps are close (within 3 seconds), group them.
    clips = []
    if trick_timestamps:
        start = trick_timestamps[0]
        end = trick_timestamps[0]

        for t in trick_timestamps[1:]:
            if t - end < MERGE_GAP_S:
                end = t  # Extend the current clip
            else:
                # Close the previous clip and start a new one
                clips.append((start, end))
                start = t
                end = t
        clips.append((start, end))  # Append the final clip

    # ------------------------------------------------------------------
    # STEP D: THE ACTION (FFmpeg Cutting)
    # ------------------------------------------------------------------
    output_dir = "highlights"
    os.makedirs(output_dir, exist_ok=True)

    highlight_artifact = wandb.Artifact("skate_highlights", type="video_collection")

    for i, (start, end) in enumerate(clips):
        # Add padding before/after the detected event
        final_start = max(0, start - CLIP_PADDING_S)
        final_end = end + CLIP_PADDING_S
        duration = final_end - final_start

        output_filename = os.path.join(output_dir, f"trick_{i+1}.mp4")

        print(f"Cutting Clip {i+1}: {final_start:.1f}s to {final_end:.1f}s")

        # FFmpeg command
        # -ss (start time), -t (duration), -c copy (fast cut, no re-encoding)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ss",
            str(final_start),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",  # Re-encode to ensure safe keyframes
            output_filename,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Add to Artifact
        highlight_artifact.add_file(output_filename)

    # ------------------------------------------------------------------
    # STEP E: PUBLISH
    # ------------------------------------------------------------------
    run.log_artifact(highlight_artifact)
    print("Highlights uploaded to W&B!")
    run.finish()


if __name__ == "__main__":
    create_highlights()
