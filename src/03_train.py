"""
Trains the "Trick Judge" -> Pushes to Registry

Refactor notes (Dec 2025):
- Uses human labels from CVAT export (intervals in frames) instead of synthetic thresholds.
- Default feature set remains "basic" (tuck_ratio only) to preserve compatibility with src/04_action.py.
- Optional "temporal" features are available for better fast-trick detection, but require updating inference code.
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
)

load_dotenv()

DEFAULT_METRICS_RUN_NAME = "summer-wave-28"
DEFAULT_ANNOTATIONS_XML = os.path.join("raw_data", "annotations.xml")
DEFAULT_LABEL = "trick"
DEFAULT_FPS = 30.0


@dataclass(frozen=True)
class Interval:
    start_frame: int
    end_frame: int  # inclusive

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame


def _load_wandb_table_as_df(table_artifact) -> pd.DataFrame:
    artifact_dir = table_artifact.download()
    json_files = [f for f in os.listdir(artifact_dir) if f.endswith(".table.json")]
    if not json_files:
        raise FileNotFoundError(f"No .table.json found in artifact dir: {artifact_dir}")
    json_path = os.path.join(artifact_dir, json_files[0])
    with open(json_path, "r") as f:
        data_dict = json.load(f)
    return pd.DataFrame(data_dict["data"], columns=data_dict["columns"])


def load_movement_metrics(
    run: wandb.sdk.wandb_run.Run, metrics_run_name: str
) -> pd.DataFrame:
    """
    Loads the `movement_metrics` W&B Table logged by `src/02_process.py`.
    """
    api = wandb.Api()
    runs = api.runs(f"{run.entity}/{run.project}")
    if not runs:
        raise RuntimeError("No W&B runs found. Run src/02_process.py first.")

    target_run = next((r for r in runs if r.name == metrics_run_name), None)
    if not target_run:
        raise RuntimeError(f"No run found with name '{metrics_run_name}'")

    artifacts = target_run.logged_artifacts()
    table_artifact = next((a for a in artifacts if "movement_metrics" in a.name), None)
    if not table_artifact:
        raise RuntimeError("No movement_metrics table found on the target run.")

    df = _load_wandb_table_as_df(table_artifact)
    if "frame" not in df.columns:
        raise ValueError("movement_metrics table is missing required column: 'frame'")
    return df


def parse_cvat_intervals(xml_path: str, label: str = DEFAULT_LABEL) -> List[Interval]:
    """
    Parses CVAT XML (v1.1) and returns inclusive frame intervals for a label.

    Supports:
    - <track label="..."> with <box frame=".." outside="0|1">
    - <tag label="..." frame=".."> (merged into consecutive intervals)
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"CVAT XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    intervals: List[Interval] = []

    # Track-based intervals (your current export format)
    for track in root.findall("track"):
        if track.attrib.get("label") != label:
            continue

        inside_frames: List[int] = []
        for box in track.findall("box"):
            frame = int(box.attrib["frame"])
            outside = box.attrib.get("outside", "0")
            if outside == "0":
                inside_frames.append(frame)

        if inside_frames:
            intervals.append(Interval(min(inside_frames), max(inside_frames)))

    # Tag-based intervals (if any)
    tag_frames: List[int] = []
    for tag_el in root.findall("tag"):
        if tag_el.attrib.get("label") == label:
            tag_frames.append(int(tag_el.attrib["frame"]))

    if tag_frames:
        tag_frames = sorted(set(tag_frames))
        start = tag_frames[0]
        prev = tag_frames[0]
        for f in tag_frames[1:]:
            if f == prev + 1:
                prev = f
                continue
            intervals.append(Interval(start, prev))
            start = prev = f
        intervals.append(Interval(start, prev))

    intervals = sorted(intervals, key=lambda x: (x.start_frame, x.end_frame))
    return intervals


def add_labels(df: pd.DataFrame, intervals: Sequence[Interval]) -> pd.DataFrame:
    """
    Adds `is_trick` label based on whether df.frame falls inside any interval.
    """
    frames = df["frame"].astype(int).to_numpy()
    y = np.zeros(len(df), dtype=np.int32)
    for i, f in enumerate(frames):
        y[i] = 1 if any(iv.contains(int(f)) for iv in intervals) else 0
    out = df.copy()
    out["is_trick"] = y
    return out


def build_features(
    df: pd.DataFrame, feature_set: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns (X, feature_names) from metrics dataframe.

    feature_set:
    - basic: tuck_ratio only (compatible with current inference script)
    - temporal: adds simple temporal features from rolling windows / diffs
    - rich: uses additional per-frame quality/body signals logged by src/02_process.py
    - rich_temporal: rich + temporal features (recommended once inference is updated)
    """
    if "tuck_ratio" not in df.columns:
        raise ValueError("movement_metrics is missing required column: 'tuck_ratio'")

    df_sorted = df.sort_values("frame").reset_index(drop=True)

    if feature_set == "basic":
        feats = ["tuck_ratio"]
        return df_sorted[feats], feats

    rich_cols = [
        # geometry / stability
        "torso_len",
        "leg_len",
        "box_area",
        "box_conf",
        "box_cx",
        "box_cy",
        # pose quality
        "mean_kpt_conf",
        "min_core_conf",
        "left_shoulder_conf",
        "right_shoulder_conf",
        "left_hip_conf",
        "right_hip_conf",
        "left_ankle_conf",
        "right_ankle_conf",
    ]

    if feature_set in {"rich", "rich_temporal"}:
        out = pd.DataFrame(index=df_sorted.index)
        # Always include the original main signal.
        out["tuck_ratio"] = df_sorted["tuck_ratio"].astype(float)
        if "hip_y" in df_sorted.columns:
            out["hip_y"] = df_sorted["hip_y"].astype(float)

        # Add any rich columns that exist in the table.
        for c in rich_cols:
            if c in df_sorted.columns:
                out[c] = pd.to_numeric(df_sorted[c], errors="coerce")
        if feature_set == "rich":
            feats = list(out.columns)
            return out, feats

        # rich_temporal = rich + rolling/delta features on core signals
        out["tuck_ratio_diff1"] = out["tuck_ratio"].diff().fillna(0.0)
        out["tuck_ratio_diff2"] = out["tuck_ratio_diff1"].diff().fillna(0.0)
        out["tuck_ratio_roll_mean3"] = (
            out["tuck_ratio"].rolling(3, min_periods=1).mean()
        )
        out["tuck_ratio_roll_std3"] = (
            out["tuck_ratio"].rolling(3, min_periods=1).std().fillna(0.0)
        )
        out["tuck_ratio_roll_mean5"] = (
            out["tuck_ratio"].rolling(5, min_periods=1).mean()
        )
        out["tuck_ratio_roll_std5"] = (
            out["tuck_ratio"].rolling(5, min_periods=1).std().fillna(0.0)
        )
        if "hip_y" in out.columns:
            out["hip_y_diff1"] = out["hip_y"].diff().fillna(0.0)
            out["hip_y_roll_mean3"] = out["hip_y"].rolling(3, min_periods=1).mean()
            out["hip_y_roll_std3"] = (
                out["hip_y"].rolling(3, min_periods=1).std().fillna(0.0)
            )

        feats = list(out.columns)
        return out, feats

    if feature_set != "temporal":
        raise ValueError(f"Unknown feature_set: {feature_set}")

    # Rolling/delta features. These are deliberately small + easy to reproduce at inference.
    out = pd.DataFrame(index=df_sorted.index)
    out["tuck_ratio"] = df_sorted["tuck_ratio"].astype(float)
    out["tuck_ratio_diff1"] = out["tuck_ratio"].diff().fillna(0.0)
    out["tuck_ratio_diff2"] = out["tuck_ratio_diff1"].diff().fillna(0.0)
    out["tuck_ratio_roll_mean3"] = out["tuck_ratio"].rolling(3, min_periods=1).mean()
    out["tuck_ratio_roll_std3"] = (
        out["tuck_ratio"].rolling(3, min_periods=1).std().fillna(0.0)
    )
    out["tuck_ratio_roll_mean5"] = out["tuck_ratio"].rolling(5, min_periods=1).mean()
    out["tuck_ratio_roll_std5"] = (
        out["tuck_ratio"].rolling(5, min_periods=1).std().fillna(0.0)
    )

    if "hip_y" in df_sorted.columns:
        out["hip_y"] = df_sorted["hip_y"].astype(float)
        out["hip_y_diff1"] = out["hip_y"].diff().fillna(0.0)
        out["hip_y_roll_mean3"] = out["hip_y"].rolling(3, min_periods=1).mean()
        out["hip_y_roll_std3"] = (
            out["hip_y"].rolling(3, min_periods=1).std().fillna(0.0)
        )

    feats = list(out.columns)
    return out, feats


def time_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time-aware split (no random leakage): last `test_frac` of frames is test.
    """
    df_sorted = df.sort_values("frame").reset_index(drop=True)
    n = len(df_sorted)
    split = int(np.floor((1.0 - test_frac) * n))
    train_idx = df_sorted.index[:split].to_numpy()
    test_idx = df_sorted.index[split:].to_numpy()
    return train_idx, test_idx


def intervals_to_table(intervals: Sequence[Interval], fps: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "start_frame": iv.start_frame,
                "end_frame": iv.end_frame,
                "start_s": iv.start_frame / fps,
                "end_s": iv.end_frame / fps,
                "duration_s": (iv.end_frame - iv.start_frame + 1) / fps,
            }
            for iv in intervals
        ]
    )


def frames_to_events(frames: Sequence[int], max_gap_frames: int = 10) -> List[Interval]:
    """
    Merge predicted-positive frames into events. Assumes frames are integers.
    """
    frames = sorted(set(int(f) for f in frames))
    if not frames:
        return []
    start = prev = frames[0]
    events: List[Interval] = []
    for f in frames[1:]:
        if f - prev <= max_gap_frames:
            prev = f
            continue
        events.append(Interval(start, prev))
        start = prev = f
    events.append(Interval(start, prev))
    return events


def interval_overlap(a: Interval, b: Interval) -> int:
    lo = max(a.start_frame, b.start_frame)
    hi = min(a.end_frame, b.end_frame)
    return max(0, hi - lo + 1)


def event_metrics(
    gt: Sequence[Interval], pred: Sequence[Interval], min_overlap_frames: int = 1
) -> dict:
    """
    Event-level precision/recall: a prediction is correct if it overlaps any GT by >= min_overlap_frames.
    """
    matched_pred = 0
    for p in pred:
        if any(interval_overlap(p, g) >= min_overlap_frames for g in gt):
            matched_pred += 1

    matched_gt = 0
    for g in gt:
        if any(interval_overlap(p, g) >= min_overlap_frames for p in pred):
            matched_gt += 1

    precision = matched_pred / len(pred) if pred else 0.0
    recall = matched_gt / len(gt) if gt else 0.0
    return {
        "event_precision": precision,
        "event_recall": recall,
        "n_gt_events": len(gt),
        "n_pred_events": len(pred),
    }


def train_model(args: argparse.Namespace) -> None:
    run = wandb.init(project="skate-cropper", job_type="train", config=vars(args))

    # Load data + labels
    df = load_movement_metrics(run, metrics_run_name=args.metrics_run_name)
    intervals = parse_cvat_intervals(args.annotations_xml, label=args.label)
    if not intervals:
        raise RuntimeError(
            f"No intervals found for label='{args.label}' in {args.annotations_xml}. "
            "Make sure you exported tracks/shapes (not just metadata)."
        )

    df = add_labels(df, intervals)

    # Optional quality filtering (helps reduce false positives from bad pose)
    if args.min_core_conf is not None and "min_core_conf" in df.columns:
        before = len(df)
        df = df[
            pd.to_numeric(df["min_core_conf"], errors="coerce") >= args.min_core_conf
        ].copy()
        run.log({"data/filtered_by_min_core_conf": before - len(df)})

    # From here onward, keep indices stable/aligned between df and feature matrices.
    df = df.sort_values("frame").reset_index(drop=True)

    # Coverage stats: how many labeled frames survived pose filtering?
    labeled = int(df["is_trick"].sum())
    run.log(
        {
            "data/rows": len(df),
            "data/pos_frames": labeled,
            "data/pos_rate": labeled / len(df) if len(df) else 0.0,
            "data/n_intervals": len(intervals),
        }
    )
    run.log(
        {
            "label_intervals": wandb.Table(
                dataframe=intervals_to_table(intervals, args.fps)
            )
        }
    )

    # Build features
    X, feature_names = build_features(df, feature_set=args.feature_set)
    if args.dropna:
        before = len(X)
        mask = ~X.isna().any(axis=1)
        # Apply the same mask to both, then reset indices to keep everything aligned.
        X = X.loc[mask].reset_index(drop=True)
        df = df.loc[mask].reset_index(drop=True)
        run.log({"data/dropped_na_rows": before - len(X)})
    y = df["is_trick"].astype(int).to_numpy()

    # Split (time-aware)
    train_idx, test_idx = time_split(df, test_frac=args.test_frac)
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    # Model choice
    if args.model == "rf":
        model = RandomForestClassifier(
            n_estimators=args.rf_estimators,
            random_state=args.seed,
            class_weight="balanced",
        )
    elif args.model == "logreg":
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=args.seed,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.fit(X_train, y_train)

    # Frame-level metrics
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= args.threshold).astype(int)
    metrics = {
        "frame/accuracy": accuracy_score(y_test, pred),
        "frame/precision": precision_score(y_test, pred, zero_division=0),
        "frame/recall": recall_score(y_test, pred, zero_division=0),
        "frame/avg_precision": (
            average_precision_score(y_test, proba)
            if len(np.unique(y_test)) > 1
            else 0.0
        ),
    }

    # Event-level metrics (on test range)
    test_frames = (
        df.sort_values("frame")
        .reset_index(drop=True)
        .iloc[test_idx]["frame"]
        .astype(int)
        .to_numpy()
    )
    pred_pos_frames = [int(f) for f, p in zip(test_frames, pred) if p == 1]
    pred_events = frames_to_events(pred_pos_frames, max_gap_frames=args.max_gap_frames)

    # GT events that overlap test window (approx): include if any labeled row in test belongs to that interval.
    test_frame_min = int(np.min(test_frames)) if len(test_frames) else 0
    test_frame_max = int(np.max(test_frames)) if len(test_frames) else 0
    gt_events_test = [
        iv
        for iv in intervals
        if interval_overlap(iv, Interval(test_frame_min, test_frame_max)) > 0
    ]
    metrics.update(
        event_metrics(
            gt_events_test, pred_events, min_overlap_frames=args.min_overlap_frames
        )
    )

    run.log(metrics)
    print(f"Model Results: {metrics}")

    # Save and register model
    model_filename = "trick_classifier.pkl"
    if args.feature_set == "basic":
        # Keep backward compatibility with existing inference that loads a bare sklearn model.
        joblib.dump(model, model_filename)
    else:
        # Store metadata for future inference refactor.
        payload = {
            "model": model,
            "feature_set": args.feature_set,
            "feature_columns": feature_names,
            "fps": args.fps,
            "threshold": args.threshold,
        }
        joblib.dump(payload, model_filename)
        print(
            "NOTE: feature_set='temporal' saves a model payload with metadata. "
            "You'll need to update src/04_action.py to compute matching features."
        )

    model_artifact = wandb.Artifact(
        name="skate_trick_classifier",
        type="model",
        description="Classifier that detects trick moments from pose-derived features",
        metadata={
            "feature_set": args.feature_set,
            "feature_columns": feature_names,
            "fps": args.fps,
            "threshold": args.threshold,
            "label": args.label,
        },
    )
    model_artifact.add_file(model_filename)
    run.log_artifact(model_artifact)
    print("Model saved and registered!")
    run.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a trick detector from pose metrics + CVAT labels."
    )
    p.add_argument("--metrics-run-name", default=DEFAULT_METRICS_RUN_NAME)
    p.add_argument("--annotations-xml", default=DEFAULT_ANNOTATIONS_XML)
    p.add_argument("--label", default=DEFAULT_LABEL)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)

    p.add_argument(
        "--feature-set",
        choices=["basic", "temporal", "rich", "rich_temporal"],
        default="basic",
        help="basic keeps inference compatibility; rich/rich_temporal require updated inference",
    )
    p.add_argument("--model", choices=["rf", "logreg"], default="rf")
    p.add_argument("--rf-estimators", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--max-gap-frames", type=int, default=10)
    p.add_argument("--min-overlap-frames", type=int, default=1)
    p.add_argument(
        "--min-core-conf",
        type=float,
        default=None,
        help="Optional: drop rows with min_core_conf below this (requires 02_process.py rich metadata).",
    )
    p.add_argument(
        "--dropna",
        action="store_true",
        help="Drop rows with NaNs in feature columns (recommended for rich features).",
    )
    return p.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
