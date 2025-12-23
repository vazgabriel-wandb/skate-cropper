# The Automation: Uses Registry Model -> Crops Video
import wandb
import cv2
import joblib
import pandas as pd
import subprocess
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()


def create_highlights():
    # 1. Initialize the "Automation" Job
    run = wandb.init(project="skate-cropper", job_type="automation")

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
    REQUIRED_STREAK = 2  # Must see 'Trick' 2 times in a row to count it
    CONFIDENCE_THRESHOLD = 0.9  # Model must be 90% sure

    # COCO Keypoint Indices
    LEFT_HIP, RIGHT_HIP = 11, 12
    LEFT_ANKLE, RIGHT_ANKLE = 15, 16
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate current time
        current_time = frame_idx / fps

        # FOCUS MODE: Only debug heavily around the trick time (8s to 12s)
        debug_mode = 8.0 < current_time < 12.0

        # Analyze every 5th frame (Matches training logic)
        if frame_idx % 5 == 0:
            results = yolo(frame, verbose=False)

            # Default assumption: No trick detected
            is_trick_frame = False

            if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
                kpts = results[0].keypoints.xyn.cpu().numpy()[0]

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
                    in_safe_zone = (
                        shoulder_y > 0.02
                        and shoulder_y < 0.98
                        and hip_y > 0.02
                        and hip_y < 0.98
                        and ankle_y > 0.02
                        and ankle_y < 0.98
                    )

                    if not in_safe_zone:
                        if debug_mode:
                            print(f"[{current_time:.2f}s] REJECTED: Touching Edge")
                        continue

                    torso_len = abs(hip_y - shoulder_y)
                    leg_len = abs(ankle_y - hip_y)

                    if torso_len > 0.01 and leg_len > 0.01:
                        tuck_ratio = leg_len / torso_len

                        # Reject impossible ratios (like 0.06)
                        if tuck_ratio <= 0.2:
                            if frame_idx % 60 == 0:
                                print(
                                    f"Frame {frame_idx}: Rejected (Impossible Ratio: {tuck_ratio:.2f})"
                                )
                            continue

                        # Prepare row for the Brain (Must match training features!)
                        df = pd.DataFrame([[tuck_ratio]], columns=["tuck_ratio"])

                        probs = model.predict_proba(df)[0]
                        trick_prob = probs[1]  # Probability of class 1 (Trick)

                        if debug_mode:
                            status = "NO"
                            if trick_prob > CONFIDENCE_THRESHOLD:
                                status = "YES"
                            print(
                                f"[{current_time:.2f}s] Ratio: {tuck_ratio:.2f} | Torso: {torso_len:.3f} | Conf: {trick_prob:.2f} -> {status}"
                            )

                        # CHANGE 2: Check Threshold
                        if trick_prob > CONFIDENCE_THRESHOLD:
                            is_trick_frame = True
                            print(
                                f"Potential detected at {frame_idx/fps:.2f}s (Conf: {trick_prob:.2f})"
                            )

            if is_trick_frame:
                consecutive_trick_frames += 1
            else:
                consecutive_trick_frames = 0  # Reset streak if we miss a frame

            # Only record timestamp if streak is met
            if consecutive_trick_frames >= REQUIRED_STREAK:
                # Avoid duplicate timestamps for the same trick
                if not trick_timestamps or (current_time - trick_timestamps[-1] > 1.0):
                    print(f"*** TRICK CONFIRMED at {current_time:.2f}s ***")
                    trick_timestamps.append(current_time)

        frame_idx += 1
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
            if t - end < 6.0:  # If less than 3 seconds since last trick frame
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
        # Add buffer (1 sec before, 1 sec after)
        final_start = max(0, start - 3.0)
        final_end = end + 3.0
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
