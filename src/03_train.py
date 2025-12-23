# Trains the "Trick Judge" -> Pushes to Registry
import wandb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from dotenv import load_dotenv
import os
import json

load_dotenv()

METRICS_RUN_NAME = "still-mountain-17"


def train_model():
    # 1. Initialize W&B Run (Job Type = Train)
    run = wandb.init(project="skate-cropper", job_type="train")

    # 2. Fetch the Data
    # We use the W&B API to get the table from your previous run.
    # We grab the most recent run in the project to save you pasting IDs.
    api = wandb.Api()
    runs = api.runs(f"{run.entity}/{run.project}")

    if not runs:
        print("No runs found! Please run 02_process.py first.")
        return

    # Get the latest run that has "movement_metrics"
    # Looking for run name = legendary-wood-10
    target_run = next((r for r in runs if r.name == METRICS_RUN_NAME), None)
    if not target_run:
        print(f"No run found with name '{METRICS_RUN_NAME}'")
        return

    # Download the table artifact (W&B stores logged tables as artifacts internally)
    # We look for the artifact named 'run-<id>-movement_metrics'
    artifacts = target_run.logged_artifacts()
    table_artifact = next((a for a in artifacts if "movement_metrics" in a.name), None)

    if not table_artifact:
        print("No movement_metrics table found in the latest run.")
        return

    artifact_dir = table_artifact.download()

    # Load the JSON table into Pandas
    # W&B Tables are stored as JSON files wrapped in a directory
    json_file = [f for f in os.listdir(artifact_dir) if f.endswith(".table.json")][0]
    json_path = os.path.join(artifact_dir, json_file)

    # Read the JSON file as a dictionary
    with open(json_path, "r") as f:
        data_dict = json.load(f)

    # Construct DataFrame from the W&B Table JSON structure
    df = pd.DataFrame(data_dict["data"], columns=data_dict["columns"])

    # 3. Create "Synthetic Labels" with NEW Threshold
    # Standing = Ratio ~1.2
    # Crouching = Ratio ~0.6
    # Let's be conservative: anything under 0.8 is a "Trick/Crouch"
    df["is_trick"] = ((df["tuck_ratio"] < 1.0) | (df["tuck_ratio"] > 3.5)).astype(int)

    print(f"Found {sum(df['is_trick'])} frames classified as 'Tricks'")

    # 4. Prepare Training Data
    features = ["tuck_ratio"]
    X = df[features]
    y = df["is_trick"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 5. Train the Brain (Random Forest)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # 6. Evaluate & Log Metrics
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
    }
    run.log(metrics)
    print(f"Model Results: {metrics}")

    # 7. Save and Register the Model
    # Save locally first
    model_filename = "trick_classifier.pkl"
    joblib.dump(model, model_filename)

    # Create a Model Artifact
    model_artifact = wandb.Artifact(
        name="skate_trick_classifier",
        type="model",
        description="Random Forest that detects jumps based on leg extension",
    )
    model_artifact.add_file(model_filename)

    # Log it! This pushes it to the Registry.
    run.log_artifact(model_artifact)

    print("Model saved and registered!")
    run.finish()


if __name__ == "__main__":
    train_model()
