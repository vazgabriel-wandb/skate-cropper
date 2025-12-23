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

    # 1. Download video
    artifact = run.use_artifact("skate_session_raw:latest")
    video_path = artifact.download() + "/skate_video.mp4"  # Adjust filename if needed

    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    data = []
    frame_idx = 0

    # COCO Keypoint Indices
    LEFT_HIP, RIGHT_HIP = 11, 12
    LEFT_ANKLE, RIGHT_ANKLE = 15, 16
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

    print("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze every 5th frame to save time
        if frame_idx % 5 == 0:
            results = model(frame, verbose=False)

            # Check if a person is detected
            if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
                kpts = results[0].keypoints.xyn.cpu().numpy()[0]

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
                        shoulder_y > 0.02
                        and shoulder_y < 0.98
                        and hip_y > 0.02
                        and hip_y < 0.98
                        and ankle_y > 0.02
                        and ankle_y < 0.98
                    )

                    if not is_in_frame:
                        # DEBUGGING: Why was it rejected?
                        # Only print occasionally to avoid spam
                        if frame_idx % 60 == 0:
                            print(
                                f"Frame {frame_idx}: Rejected (Touching Edge) - Hip Y: {hip_y:.2f}"
                            )
                        continue

                    # 2. Calculate Lengths
                    torso_len = abs(hip_y - shoulder_y)
                    leg_len = abs(ankle_y - hip_y)

                    # 3. Calculate "Tuck Ratio" (Scale Invariant!)
                    # Avoid division by zero
                    if torso_len > 0.02:
                        tuck_ratio = leg_len / torso_len

                        # --- NEW GUARDRAIL: HUMAN LIMIT ---
                        # Reject the 0.06 "ghost" frames here too!
                        if tuck_ratio > 0.2:
                            # DEBUGGING: Print interesting frames
                            # If ratio is low (trick-like), show us why
                            if tuck_ratio < 0.8:
                                print(
                                    f"Frame {frame_idx} ({frame_idx/fps:.1f}s): VALID Trick Ratio {tuck_ratio:.2f}"
                                )

                            data.append(
                                {
                                    "frame": frame_idx,
                                    "hip_y": hip_y,
                                    "tuck_ratio": tuck_ratio,
                                }
                            )
                        else:
                            # Optional: Print rejected ghosts
                            if frame_idx % 60 == 0:
                                print(
                                    f"Frame {frame_idx}: Rejected (Impossible Ratio: {tuck_ratio:.2f})"
                                )

        frame_idx += 1

    cap.release()

    # 3. Log to W&B
    df = pd.DataFrame(data)

    # Create the artifacts if data exists
    if not df.empty:
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
