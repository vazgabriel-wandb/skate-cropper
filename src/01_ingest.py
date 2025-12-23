# Uploads raw video to W&B Artifacts
import wandb
import os
import sys
from dotenv import load_dotenv

# Load secrets from .env
load_dotenv()

def upload_raw_footage(file_path):
    # 1. Initialize the W&B Run
    # job_type="ingest" helps you filter runs later in the dashboard
    run = wandb.init(
        job_type="ingest", notes="Uploading raw skate session for analysis"
    )

    # 2. Create an Artifact
    # Type "raw-footage" is key. Your automation will listen for this specific type.
    artifact = wandb.Artifact(
        name="skate_session_raw",
        type="raw-footage",
        description="Uncut video of skate session",
    )

    # 3. Add the file to the artifact
    if os.path.exists(file_path):
        print(f"Adding {file_path} to artifact...")
        artifact.add_file(file_path)
    else:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    # 4. Log (Upload) the artifact
    # This triggers the upload to W&B cloud.
    run.log_artifact(artifact)

    print("Upload complete!")
    run.finish()


if __name__ == "__main__":
    # Usage: python src/01_ingest.py path/to/video.mp4
    if len(sys.argv) < 2:
        print("Please provide the path to the video file.")
    else:
        upload_raw_footage(sys.argv[1])
