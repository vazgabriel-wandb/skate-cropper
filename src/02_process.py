# Downloads video -> Runs YOLO -> Saves numeric data
import wandb
from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def extract_features():
    run = wandb.init(job_type="feature-extraction", project="skate-cropper")

    # ----------------------------
    # Config knobs (W&B)
    # ----------------------------
    default_cfg = {
        # sampling: 2 => ~15fps on 30fps
        "sample_stride": 2,
        # pose filtering
        "edge_margin": 0.02,
        "min_torso_len": 0.02,
        "min_tuck_ratio": 0.2,
        # debug printing cadence (in frames, original frame indices)
        "debug_print_every_n_frames": 60,
    }
    run.config.update(default_cfg, allow_val_change=True)

    # 1. Download video
    artifact = run.use_artifact("skate_session_raw:latest")
    video_path = artifact.download() + "/skate_video.mp4"  # Adjust filename if needed

    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    data = []
    frame_idx = 0

    # Read knobs from config
    sample_stride = int(run.config.get("sample_stride", 2))
    edge_margin = float(run.config.get("edge_margin", 0.02))
    min_torso_len = float(run.config.get("min_torso_len", 0.02))
    min_tuck_ratio = float(run.config.get("min_tuck_ratio", 0.2))
    debug_every = int(run.config.get("debug_print_every_n_frames", 60))

    # COCO Keypoint Indices
    LEFT_HIP, RIGHT_HIP = 11, 12
    LEFT_ANKLE, RIGHT_ANKLE = 15, 16
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

    print("Processing video...")

    # Track filtering/quality stats for debugging + W&B logging
    stats = {
        "frames_total": 0,
        "frames_sampled": 0,
        "no_person": 0,
        "missing_boxes": 0,
        "invalid_keypoints": 0,
        "touching_edge": 0,
        "torso_too_small": 0,
        "impossible_ratio": 0,
        "logged": 0,
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        stats["frames_total"] += 1
        # IMPORTANT: increment the frame counter for every decoded frame.
        # (Avoid `continue` paths skipping the increment and causing an infinite loop.)
        current_frame_idx = frame_idx
        frame_idx += 1

        # Analyze every Nth frame to save time
        if current_frame_idx % sample_stride == 0:
            stats["frames_sampled"] += 1
            results = model(frame, verbose=False)

            # Check if a person is detected
            if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
                # Robustly pick the "main" person:
                # choose the detection with highest (box_conf * box_area) when boxes exist,
                # else fall back to the first keypoint set.
                chosen_idx = 0
                try:
                    boxes = results[0].boxes
                    if (
                        boxes is not None
                        and boxes.xyxy is not None
                        and len(boxes.xyxy) > 0
                    ):
                        xyxy = boxes.xyxy.cpu().numpy()
                        conf = (
                            boxes.conf.cpu().numpy()
                            if boxes.conf is not None
                            else np.ones(len(xyxy))
                        )
                        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                        score = conf * areas
                        chosen_idx = int(np.argmax(score))
                    else:
                        stats["missing_boxes"] += 1
                except Exception:
                    stats["missing_boxes"] += 1

                kpts = results[0].keypoints.xyn.cpu().numpy()[chosen_idx]
                # Keypoint confidences (0..1). Shape: (K,)
                kpts_conf = None
                if getattr(results[0].keypoints, "conf", None) is not None:
                    kpts_conf = results[0].keypoints.conf.cpu().numpy()[chosen_idx]

                if (
                    kpts[LEFT_SHOULDER][0] > 0
                    and kpts[LEFT_HIP][0] > 0
                    and kpts[LEFT_ANKLE][0] > 0
                ):

                    # 1. Get Y-coordinates
                    shoulder_y = (kpts[LEFT_SHOULDER][1] + kpts[RIGHT_SHOULDER][1]) / 2
                    hip_y = (kpts[LEFT_HIP][1] + kpts[RIGHT_HIP][1]) / 2
                    ankle_y = (kpts[LEFT_ANKLE][1] + kpts[RIGHT_ANKLE][1]) / 2

                    # 2. EDGE CHECK (The Fix)
                    # Reject if any part is too close to Top (0.0) or Bottom (1.0)
                    # We use a 2% margin (0.02)
                    is_in_frame = (
                        shoulder_y > edge_margin
                        and shoulder_y < 1.0 - edge_margin
                        and hip_y > edge_margin
                        and hip_y < 1.0 - edge_margin
                        and ankle_y > edge_margin
                        and ankle_y < 1.0 - edge_margin
                    )

                    if not is_in_frame:
                        stats["touching_edge"] += 1
                        # DEBUGGING: Why was it rejected?
                        # Only print occasionally to avoid spam
                        if current_frame_idx % debug_every == 0:
                            print(
                                f"Frame {current_frame_idx}: Rejected (Touching Edge) - Hip Y: {hip_y:.2f}"
                            )
                        continue

                    # 2. Calculate Lengths
                    torso_len = abs(hip_y - shoulder_y)
                    leg_len = abs(ankle_y - hip_y)

                    # 3. Calculate "Tuck Ratio" (Scale Invariant!)
                    # Avoid division by zero
                    if torso_len > min_torso_len:
                        tuck_ratio = float(leg_len / torso_len)

                        # --- NEW GUARDRAIL: HUMAN LIMIT ---
                        # Reject the 0.06 "ghost" frames here too!
                        if tuck_ratio > min_tuck_ratio:
                            # DEBUGGING: Print interesting frames
                            # If ratio is low (trick-like), show us why
                            if tuck_ratio < 0.8:
                                print(
                                    f"Frame {current_frame_idx} ({current_frame_idx/fps:.1f}s): VALID Trick Ratio {tuck_ratio:.2f}"
                                )

                            # Optional: attach quality features for training/debugging
                            left_shoulder_conf = (
                                float(kpts_conf[LEFT_SHOULDER])
                                if kpts_conf is not None
                                else np.nan
                            )
                            right_shoulder_conf = (
                                float(kpts_conf[RIGHT_SHOULDER])
                                if kpts_conf is not None
                                else np.nan
                            )
                            left_hip_conf = (
                                float(kpts_conf[LEFT_HIP])
                                if kpts_conf is not None
                                else np.nan
                            )
                            right_hip_conf = (
                                float(kpts_conf[RIGHT_HIP])
                                if kpts_conf is not None
                                else np.nan
                            )
                            left_ankle_conf = (
                                float(kpts_conf[LEFT_ANKLE])
                                if kpts_conf is not None
                                else np.nan
                            )
                            right_ankle_conf = (
                                float(kpts_conf[RIGHT_ANKLE])
                                if kpts_conf is not None
                                else np.nan
                            )
                            mean_kpt_conf = (
                                float(np.nanmean(kpts_conf))
                                if kpts_conf is not None
                                else np.nan
                            )
                            min_core_conf = (
                                float(
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
                                if kpts_conf is not None
                                else np.nan
                            )

                            # Bounding box stats if available
                            box_area = np.nan
                            box_conf = np.nan
                            box_cx = np.nan
                            box_cy = np.nan
                            try:
                                boxes = results[0].boxes
                                if (
                                    boxes is not None
                                    and boxes.xyxy is not None
                                    and len(boxes.xyxy) > chosen_idx
                                ):
                                    xyxy = boxes.xyxy.cpu().numpy()[chosen_idx]
                                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                                    box_area = float(
                                        max(0.0, x2 - x1) * max(0.0, y2 - y1)
                                    )
                                    if (
                                        boxes.conf is not None
                                        and len(boxes.conf) > chosen_idx
                                    ):
                                        box_conf = float(
                                            boxes.conf.cpu().numpy()[chosen_idx]
                                        )
                                    box_cx = float((x1 + x2) / 2.0)
                                    box_cy = float((y1 + y2) / 2.0)
                            except Exception:
                                pass

                            data.append(
                                {
                                    "frame": current_frame_idx,
                                    "hip_y": hip_y,
                                    "tuck_ratio": tuck_ratio,
                                    # Extra metadata (optional for future training)
                                    "fps": float(fps) if fps else np.nan,
                                    "sample_stride": int(sample_stride),
                                    "chosen_person_idx": int(chosen_idx),
                                    "torso_len": float(torso_len),
                                    "leg_len": float(leg_len),
                                    "shoulder_y": float(shoulder_y),
                                    "hip_y_raw": float(hip_y),
                                    "ankle_y": float(ankle_y),
                                    "edge_margin": float(edge_margin),
                                    "left_shoulder_conf": left_shoulder_conf,
                                    "right_shoulder_conf": right_shoulder_conf,
                                    "left_hip_conf": left_hip_conf,
                                    "right_hip_conf": right_hip_conf,
                                    "left_ankle_conf": left_ankle_conf,
                                    "right_ankle_conf": right_ankle_conf,
                                    "mean_kpt_conf": mean_kpt_conf,
                                    "min_core_conf": min_core_conf,
                                    "box_area": box_area,
                                    "box_conf": box_conf,
                                    "box_cx": box_cx,
                                    "box_cy": box_cy,
                                }
                            )
                            stats["logged"] += 1
                        else:
                            stats["impossible_ratio"] += 1
                            # Optional: Print rejected ghosts
                            if current_frame_idx % debug_every == 0:
                                print(
                                    f"Frame {current_frame_idx}: Rejected (Impossible Ratio: {tuck_ratio:.2f})"
                                )
                    else:
                        stats["torso_too_small"] += 1
                else:
                    stats["invalid_keypoints"] += 1
            else:
                stats["no_person"] += 1

    cap.release()

    # 3. Log to W&B
    df = pd.DataFrame(data)

    # Create the artifacts if data exists
    if not df.empty:
        # Log run config + stats for reproducibility/debugging
        run.config.update(
            {
                "video_fps": float(fps) if fps else None,
                "sample_stride": int(sample_stride),
                "effective_fps": float(fps / sample_stride) if fps else None,
                "edge_margin": 0.02,
            },
            allow_val_change=True,
        )
        run.log({f"filter/{k}": v for k, v in stats.items()})

        table = wandb.Table(dataframe=df)
        run.log(
            {
                "movement_metrics": table,
                "ratio_plot": wandb.plot.line(
                    table, "frame", "tuck_ratio", title="Leg/Torso Ratio (Cleaned)"
                ),
            }
        )
        print(f"Logged {len(df)} valid frames.")
    else:
        print("Warning: No valid frames found! Check thresholds.")

    run.finish()


if __name__ == "__main__":
    extract_features()
