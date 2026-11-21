#!/usr/bin/env python3
"""Train and evaluate the ear-feature LightGBM model with subject-wise CV."""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nyas-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ear_training_data import FEATURE_COLUMNS, process_one_night  # noqa: E402

INPUT = ROOT / "data/processed/ear_sleep_training/ear_sleep_training.csv"
MANIFEST = ROOT / "data/processed/ear_sleep_training/manifest.json"
YASA_BASELINE = ROOT / "results/yasa/sub-016_ses-001_comparison.json"
OUT = ROOT / "results/ear_lightgbm"
KEYS = ["subject", "session", "epoch_index"]
LABEL_CODES = {1: "Wake", 3: "N1", 4: "N2", 5: "N3", 2: "REM"}
CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
CONTEXT_LAGS = (1, 2, 4)
ROLLING_WINDOW = 5
SEED = 42
PARAMS = {
    "objective": "multiclass",
    "learning_rate": 0.05,
    "n_estimators": 400,
    "num_leaves": 15,
    "min_child_samples": 50,
    "colsample_bytree": 0.7,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": SEED,
    "verbosity": -1,
    "n_jobs": -1,
}


def context_base_features() -> list[str]:
    return [
        name for name in FEATURE_COLUMNS
        if name.endswith("_relative_power")
        or name.endswith("_ratio")
        or name.startswith("hjorth_")
        or name == "std_uv"
    ]


def robust_scale_by_night(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Robust-scale available rows independently within each recording night."""
    scaled = frame.copy()
    for _, indexes in scaled.groupby(["subject", "session"], sort=False).groups.items():
        values = scaled.loc[indexes, feature_names].astype(float)
        medians = values.median(axis=0)
        iqrs = values.quantile(0.75) - values.quantile(0.25)
        iqrs = iqrs.mask(iqrs == 0, 1.0)
        scaled.loc[indexes, feature_names] = (values - medians) / iqrs
    return scaled


def add_context_features(frame: pd.DataFrame, base_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Add exact epoch-index lags/leads and centered five-epoch summaries."""
    result = frame.copy()
    context_names = [
        name
        for feature in base_features
        for name in (
            *(f"{feature}_{side}{distance}" for distance in CONTEXT_LAGS for side in ("lag", "lead")),
            f"{feature}_roll5_mean",
            f"{feature}_roll5_std",
        )
    ]
    for _, group in result.groupby(["subject", "session"], sort=False):
        ordered = group.sort_values("epoch_index")
        row_indexes = ordered.index
        epoch_index = ordered["epoch_index"].astype(int).to_numpy()
        indexed = ordered.set_index("epoch_index")
        full_index = pd.RangeIndex(int(epoch_index.min()), int(epoch_index.max()) + 1)
        for feature in base_features:
            series = indexed[feature].astype(float)
            full_series = series.reindex(full_index)
            for distance in CONTEXT_LAGS:
                lag_name = f"{feature}_lag{distance}"
                lead_name = f"{feature}_lead{distance}"
                result.loc[row_indexes, lag_name] = [
                    series.get(int(epoch) - distance, np.nan) for epoch in epoch_index
                ]
                result.loc[row_indexes, lead_name] = [
                    series.get(int(epoch) + distance, np.nan) for epoch in epoch_index
                ]
            roll_mean = full_series.rolling(ROLLING_WINDOW, center=True, min_periods=ROLLING_WINDOW).mean()
            roll_std = full_series.rolling(ROLLING_WINDOW, center=True, min_periods=ROLLING_WINDOW).std(ddof=0)
            mean_name = f"{feature}_roll5_mean"
            std_name = f"{feature}_roll5_std"
            result.loc[row_indexes, mean_name] = roll_mean.reindex(epoch_index).to_numpy()
            result.loc[row_indexes, std_name] = roll_std.reindex(epoch_index).to_numpy()
    return result, context_names


def make_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    scaled = robust_scale_by_night(frame, list(FEATURE_COLUMNS))
    with_context, context_names = add_context_features(scaled, context_base_features())
    return with_context, list(FEATURE_COLUMNS) + context_names


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    return {
        "n_epochs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)), average="macro", zero_division=0)),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                  "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_counts": confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES))).tolist(),
    }


def smooth_by_night(predictions: pd.DataFrame, pred_column: str, window: int) -> np.ndarray:
    """Median-filter predictions within contiguous epoch runs of each night."""
    output = np.empty(len(predictions), dtype=np.int64)
    output.fill(-1)
    for _, group in predictions.groupby(["subject", "session"], sort=False):
        ordered = group.sort_values("epoch_index")
        positions = predictions.index.get_indexer(ordered.index)
        epochs = ordered["epoch_index"].to_numpy(dtype=int)
        values = ordered[pred_column].to_numpy(dtype=int)
        cuts = np.flatnonzero(np.diff(epochs) != 1) + 1
        for pos_block, value_block in zip(np.split(positions, cuts), np.split(values, cuts)):
            output[pos_block] = median_filter(value_block, size=window, mode="nearest")
    if np.any(output < 0):
        raise ValueError("Smoothing failed to assign predictions to every epoch")
    return output


def subject_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = {
        "raw": "pred_id",
        "median_3": "pred_smooth_3_id",
        "median_5": "pred_smooth_5_id",
    }
    for method, column in methods.items():
        for subject, group in predictions.groupby("subject", sort=True):
            scores = metric_bundle(group.true_id.to_numpy(), group[column].to_numpy())
            rows.append({"prediction_set": method, "subject": subject,
                         "accuracy": scores["accuracy"], "cohen_kappa": scores["cohen_kappa"],
                         "macro_f1": scores["macro_f1"], "n_epochs": scores["n_epochs"]})
        subject_rows = [row for row in rows if row["prediction_set"] == method]
        for aggregate, fn in (("MEAN", np.mean), ("STD", np.std)):
            rows.append({
                "prediction_set": method,
                "subject": aggregate,
                "accuracy": float(fn([row["accuracy"] for row in subject_rows])),
                "cohen_kappa": float(fn([row["cohen_kappa"] for row in subject_rows])),
                "macro_f1": float(fn([row["macro_f1"] for row in subject_rows])),
                "n_epochs": float(fn([row["n_epochs"] for row in subject_rows])),
            })
    return pd.DataFrame(rows)


def save_confusion_plot(predictions: pd.DataFrame) -> dict:
    specs = [("Raw", "pred_id"), ("Median filter (3)", "pred_smooth_3_id"),
             ("Median filter (5)", "pred_smooth_5_id")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    matrices = {}
    for axis, (title, column) in zip(axes, specs):
        y = predictions.true_id.to_numpy()
        pred = predictions[column].to_numpy()
        counts = confusion_matrix(y, pred, labels=np.arange(len(CLASS_NAMES)))
        normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        matrices[column] = {"counts": counts.tolist(), "row_normalized": normalized.tolist()}
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        axis.set_title(title)
        axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                axis.text(j, i, f"{counts[i, j]}\n{normalized[i,j]:.2f}",
                          ha="center", va="center", fontsize=7,
                          color="white" if normalized[i, j] > 0.55 else "black")
    fig.colorbar(image, ax=axes, shrink=0.78, label="Row-normalized fraction")
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    return matrices


def save_importance_plot(importances: pd.DataFrame) -> list[dict]:
    mean_gain = importances.mean(axis=0).sort_values(ascending=False)
    top = mean_gain.head(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.barh(top.index, top.values, color="#3973ac")
    ax.set_xlabel("Mean gain across LOSO folds")
    ax.set_title("Top 15 LightGBM features")
    fig.savefig(OUT / "feature_importance.png", dpi=180)
    plt.close(fig)
    return [{"feature": name, "mean_gain": float(value)}
            for name, value in mean_gain.head(15).items()]


def build_heldout_features(feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, labels = process_one_night("sub-016", "001")
    if not features[KEYS].equals(labels[KEYS]):
        raise ValueError("Held-out feature and label keys differ")
    merged = features.merge(labels[KEYS + ["stage_code", "stage_name"]],
                            on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(features) or len(merged) != len(labels):
        raise ValueError("Held-out feature/label join lost rows")
    transformed, columns = make_features(merged)
    if columns != feature_names:
        raise ValueError("Held-out feature schema differs from training")
    valid_labels = merged.stage_code.isin(LABEL_CODES)
    heldout = transformed.loc[valid_labels].copy()
    if len(heldout) != 690:
        raise ValueError(f"Expected 690 labeled sub-016 epochs to match YASA; got {len(heldout)}")
    return transformed, heldout


def main() -> None:
    if not INPUT.exists() or not MANIFEST.exists():
        raise FileNotFoundError("Build the cohort table first: " + str(INPUT))
    if not YASA_BASELINE.exists():
        raise FileNotFoundError("Stored YASA baseline is missing: " + str(YASA_BASELINE))
    OUT.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT).sort_values(KEYS).reset_index(drop=True)
    manifest = json.loads(MANIFEST.read_text())
    if not set(KEYS + ["stage_code", "stage_name", "signal_valid"]).issubset(data.columns):
        raise ValueError("Cohort table is missing required keys, labels, or signal flags")
    if not set(FEATURE_COLUMNS).issubset(data.columns):
        raise ValueError("Cohort table feature schema differs from ear_training_data.FEATURE_COLUMNS")
    if data.duplicated(KEYS).any():
        raise ValueError("Duplicate epoch keys found in the combined cohort table")
    numeric_values = data[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("NaN or infinity found in cohort feature columns")
    if not data.stage_code.isin(LABEL_CODES).all():
        raise ValueError("Unexpected label code in already-filtered cohort table")
    if not data.signal_valid.astype(bool).all():
        warnings.warn("Input contains signal-invalid rows; they will be removed after context generation")

    OUT.mkdir(parents=True, exist_ok=True)
    transformed, model_features = make_features(data)
    keep = transformed.stage_code.isin(LABEL_CODES) & transformed.signal_valid.astype(bool)
    filtered = transformed.loc[keep].copy().sort_values(KEYS).reset_index(drop=True)
    filtered["true_id"] = filtered.stage_name.map(LABEL_TO_ID)
    if filtered.true_id.isna().any():
        raise ValueError("Stage name could not be mapped to the fixed five-class encoding")
    filtered["true_id"] = filtered.true_id.astype(int)
    if filtered.groupby("subject").true_id.nunique().lt(len(CLASS_NAMES)).any():
        raise ValueError("A held-out subject's training fold would be missing at least one class")
    X = filtered[model_features].astype(float)
    y = filtered.true_id.to_numpy(dtype=int)
    groups = filtered.subject.to_numpy()
    logo = LeaveOneGroupOut()
    oof_proba = np.full((len(filtered), len(CLASS_NAMES)), np.nan, dtype=float)
    fold_importances = []
    fold_subjects = []
    for train_pos, test_pos in logo.split(X, y, groups):
        held_subject = str(groups[test_pos[0]])
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X.iloc[train_pos], y[train_pos])
        probs = model.predict_proba(X.iloc[test_pos])
        oof_proba[test_pos] = probs[:, np.asarray(model.classes_, dtype=int)]
        fold_importances.append(pd.Series(
            model.booster_.feature_importance(importance_type="gain"), index=model_features
        ))
        fold_subjects.append(held_subject)
        print(f"Completed LOSO fold for {held_subject}: {len(test_pos)} epochs", flush=True)
    if not np.isfinite(oof_proba).all():
        raise ValueError("LOSO did not produce a finite probability for every retained epoch")

    preds = filtered[KEYS + ["stage_name", "true_id"]].copy()
    preds = preds.rename(columns={"stage_name": "true"})
    preds["pred_id"] = np.argmax(oof_proba, axis=1)
    preds["pred"] = preds.pred_id.map(ID_TO_LABEL)
    for class_id, name in ID_TO_LABEL.items():
        preds[f"prob_{name.lower()}"] = oof_proba[:, class_id]
    preds["pred_smooth_3_id"] = smooth_by_night(preds, "pred_id", 3)
    preds["pred_smooth_5_id"] = smooth_by_night(preds, "pred_id", 5)
    preds["pred_smooth_3"] = preds.pred_smooth_3_id.map(ID_TO_LABEL)
    preds["pred_smooth_5"] = preds.pred_smooth_5_id.map(ID_TO_LABEL)
    metric_columns = {"raw": "pred_id", "median_3": "pred_smooth_3_id", "median_5": "pred_smooth_5_id"}
    overall = {
        name: metric_bundle(preds.true_id.to_numpy(), preds[column].to_numpy())
        for name, column in metric_columns.items()
    }

    per_subject = subject_metric_table(preds)
    per_subject.to_csv(OUT / "per_subject_metrics.csv", index=False)
    prediction_columns = KEYS + ["true", "pred", "pred_smooth_3", "pred_smooth_5"] + [
        f"prob_{name.lower()}" for name in CLASS_NAMES
    ]
    preds[prediction_columns].rename(columns={"epoch_index": "epoch"}).to_csv(
        OUT / "oof_predictions.csv", index=False
    )

    matrices = save_confusion_plot(preds)
    importance = pd.concat(fold_importances, axis=1).T
    top_importance = save_importance_plot(importance)
    top3 = [item["feature"] for item in top_importance[:3]]
    amplitude_tokens = ("mean_uv", "median_uv", "std_uv", "rms_uv", "peak_to_peak_uv", "abs_q95_uv", "power_uv2")
    risky_top3 = [name for name in top3 if any(token in name for token in amplitude_tokens)]
    if risky_top3:
        warnings.warn("Raw-amplitude feature(s) appear in the top three importances: " + ", ".join(risky_top3))

    # Fit the specified all-cohort model for the independent sub-016 night.
    heldout_all, heldout = build_heldout_features(model_features)
    final_model = lgb.LGBMClassifier(**PARAMS)
    final_model.fit(X, y)
    heldout_X = heldout[model_features].astype(float)
    heldout_y = heldout.stage_name.map(LABEL_TO_ID).to_numpy(dtype=int)
    heldout_probs = final_model.predict_proba(heldout_X)[:, np.asarray(final_model.classes_, dtype=int)]
    heldout_pred = np.argmax(heldout_probs, axis=1)
    heldout_metrics = metric_bundle(heldout_y, heldout_pred)
    baseline = json.loads(YASA_BASELINE.read_text())
    if baseline["recording"] != "sub-016:001":
        raise ValueError("YASA comparison JSON is not the requested sub-016:001 baseline")
    ear_yasa = baseline["signals"]["in_ear"]["summary"]
    scalp_yasa = baseline["signals"]["scalp_psg"]["summary"]
    if ear_yasa["epochs_compared"] != 690 or scalp_yasa["epochs_compared"] != 690:
        raise ValueError("Stored YASA baseline does not use the requested 690 epochs")
    comparison = [
        {"model": "YASA pretrained", "signal": "In-ear ELE−ERE", "epochs": 690,
         "accuracy": ear_yasa["accuracy_percent"] / 100, "error": ear_yasa["error_percent"] / 100,
         "source": "existing results/yasa JSON"},
        {"model": "LightGBM", "signal": "In-ear ELE−ERE", "epochs": 690,
         "accuracy": heldout_metrics["accuracy"], "error": 1 - heldout_metrics["accuracy"],
         "source": "sub-016:001 held out from cohort training"},
        {"model": "YASA pretrained", "signal": "Scalp C3−M2", "epochs": 690,
         "accuracy": scalp_yasa["accuracy_percent"] / 100, "error": scalp_yasa["error_percent"] / 100,
         "source": "existing results/yasa JSON"},
    ]

    night_drop_counts = {}
    for recording, info in manifest.get("completed_nights", {}).items():
        labels = info.get("label_counts", {})
        artifact_labels = sum(int(labels.get(k, 0)) for k in ("A", "Artefact", "Unscored"))
        night_drop_counts[recording] = {
            "source_epochs": info.get("source_epochs", 0),
            "artifact_or_unscored_label_epochs_dropped_during_cohort_build": artifact_labels,
            "signal_invalid_epochs_dropped_during_cohort_build": info.get("unusable_signal_epochs_dropped", 0),
            "usable_epochs_in_model_input": int((filtered.subject.eq(recording.split(":")[0]) &
                                                   filtered.session.astype(str).str.zfill(3).eq(recording.split(":")[1])).sum()),
            "cohort_build_status": info.get("status"),
        }
    run_config = {
        "input_csv": str(INPUT.relative_to(ROOT)),
        "cohort_manifest": str(MANIFEST.relative_to(ROOT)),
        "cohort_subjects": sorted(filtered.subject.unique().tolist()),
        "heldout_recording": "sub-016:001",
        "heldout_epochs": int(len(heldout)),
        "label_encoding": {name: int(value) for name, value in LABEL_TO_ID.items()},
        "params": PARAMS,
        "random_seed": SEED,
        "base_feature_columns": list(FEATURE_COLUMNS),
        "context_base_features": context_base_features(),
        "context_lags_and_leads_epochs": list(CONTEXT_LAGS),
        "context_rolling_window_epochs": ROLLING_WINDOW,
        "model_feature_columns": model_features,
        "time_in_night_enabled": False,
        "per_night_scaling": "median/IQR on rows present in the cohort CSV",
        "artifact_epoch_limitation": (
            "The cohort builder had already removed artifact/unscored and signal-invalid epochs. "
            "They are therefore unavailable to the per-night scaler; counts are recorded from its manifest."
        ),
        "dropped_epoch_counts_by_night": night_drop_counts,
        "fold_subjects": fold_subjects,
        "cohort_rows_used": int(len(filtered)),
        "cohort_rows_available": int(len(data)),
        "warning_top3_amplitude_features": risky_top3,
    }
    (OUT / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    heldout_table = pd.DataFrame(comparison)
    report = [
        "# Ear EEG LightGBM evaluation",
        "",
        "## Summary",
        "",
        f"Leave-one-subject-out evaluation covered {len(filtered):,} retained epochs from "
        f"{filtered.subject.nunique()} training subjects across {len(fold_subjects)} folds.",
        f"Raw LOSO accuracy was {overall['raw']['accuracy']:.1%}, Cohen's kappa "
        f"{overall['raw']['cohen_kappa']:.3f}, and macro-F1 {overall['raw']['macro_f1']:.3f}.",
        f"The 3-epoch median result was accuracy {overall['median_3']['accuracy']:.1%}, "
        f"macro-F1 {overall['median_3']['macro_f1']:.3f}; the 5-epoch result was "
        f"accuracy {overall['median_5']['accuracy']:.1%}, macro-F1 {overall['median_5']['macro_f1']:.3f}.",
        "",
        "## Generalization across subjects",
        "",
        "Per-subject metrics and mean/std rows are in `per_subject_metrics.csv`.",
        "",
        "## Weakest stages and class metrics (raw LOSO)",
        "",
        "| Stage | Precision | Recall | F1 | Epochs |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, values in overall["raw"]["per_class"].items():
        report.append(f"| {stage} | {values['precision']:.3f} | {values['recall']:.3f} | {values['f1']:.3f} | {values['support']} |")
    weakest = min(overall["raw"]["per_class"].items(), key=lambda item: item[1]["f1"])[0]
    report += [
        "",
        f"The weakest raw-prediction class by F1 was **{weakest}**.",
        "",
        "## Most important features",
        "",
        "| Rank | Feature | Mean gain |",
        "|---:|---|---:|",
    ]
    report.extend(f"| {i} | {x['feature']} | {x['mean_gain']:.3f} |" for i, x in enumerate(top_importance, 1))
    if risky_top3:
        report += ["", "**Warning:** a time-based or raw-amplitude feature appears in the top three: " + ", ".join(risky_top3) + "."]
    report += [
        "",
        "## Held-out comparison: sub-016 session 001",
        "",
        "All rows below use the same 690 Scoring1-labeled epochs as the stored YASA baseline. YASA results were read from the existing JSON and not recomputed.",
        "",
        "| Model | Signal | Epochs | Accuracy | Error |",
        "|---|---|---:|---:|---:|",
    ]
    report.extend(f"| {row['model']} | {row['signal']} | {row['epochs']} | {row['accuracy']:.1%} | {row['error']:.1%} |" for row in comparison)
    report += [
        "",
        "No cohort-wide YASA scores are available; only sub-016 session 001 has stored YASA results.",
        "",
        "## Caveats",
        "",
        "- Four of the 40 selected nights were excluded: three had no finite ear-channel data and one had a mismatched EEGLAB header/data file. Several retained nights have high signal-invalid rates; see `run_config.json` for per-night counts.",
        "- The cohort CSV was already filtered by the dataset builder. Artifact, unscored, and signal-invalid rows were unavailable when calculating per-night robust scaling and context features; available rows were scaled before the model-stage label filter.",
        "- Centered windows and lead features use future epochs. These results describe offline staging, not real-time prediction.",
        "- The sub-016 evaluation is a single held-out night and should be treated as a demonstration, not an independent validation study.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(report))
    metrics = {
        "cohort_loso": overall,
        "per_subject_mean_std": {
            method: {
                metric: {
                    "mean": float(per_subject.loc[
                        (per_subject.prediction_set == method) & (~per_subject.subject.isin(["MEAN", "STD"])), metric
                    ].mean()),
                    "std": float(per_subject.loc[
                        (per_subject.prediction_set == method) & (~per_subject.subject.isin(["MEAN", "STD"])), metric
                    ].std(ddof=0)),
                }
                for metric in ("accuracy", "cohen_kappa", "macro_f1")
            }
            for method in metric_columns
        },
        "confusion_matrices": matrices,
        "top_15_feature_importance_mean_gain": top_importance,
        "heldout_sub-016_ses-001": heldout_metrics,
        "baseline_comparison": comparison,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Training complete. Results: {OUT.relative_to(ROOT)}", flush=True)
    print(json.dumps({"raw_loso": overall["raw"], "heldout_sub-016": heldout_metrics}, indent=2))


if __name__ == "__main__":
    main()
