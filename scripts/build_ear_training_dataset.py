#!/usr/bin/env python3
"""Download, process, and release each selected PSG night, then stack its epochs."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ear_training_data import CONFIG, FEATURE_COLUMNS, process_one_night
from fetch_data import download, inventory

SUBJECTS = (
    "sub-001", "sub-002", "sub-003", "sub-004", "sub-008",
    "sub-009", "sub-011", "sub-012", "sub-017", "sub-018",
)
SESSIONS = ("001", "002", "003", "004")
# EEGLAB header and paired data file disagree for this recording; it cannot be read reliably.
SKIPPED_RECORDINGS = {
    "sub-011:004": "EEGLAB header expects 1,740,387,000 bytes; paired .fdt is 309,815,000 bytes",
}
OUTPUT = ROOT / "data" / "processed" / "ear_sleep_training"
BATCHES = OUTPUT / "by_night"
MANIFEST_PATH = OUTPUT / "manifest.json"
FINAL_PATH = OUTPUT / "ear_sleep_training.csv"
KEY_COLUMNS = ["subject", "session", "epoch_index"]
LABEL_COLUMNS = ["stage_code", "stage_name", "is_sleep_stage"]


def write_manifest(manifest: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(MANIFEST_PATH)


def object_keys(subject: str, session: str) -> tuple[str, str, str]:
    stem = f"{subject}_ses-{session}_task-sleep_acq-PSG_eeg"
    prefix = f"ds005185/{subject}/ses-{session}/eeg/"
    return (
        prefix + stem + ".set",
        prefix + stem + ".fdt",
        prefix + f"{subject}_ses-{session}_task-sleep_acq-scoring1_events.tsv",
    )


def local_targets(subject: str, session: str) -> tuple[Path, Path, Path]:
    stem = f"{subject}_ses-{session}_task-sleep_acq-PSG_eeg"
    return (
        ROOT / "data/raw" / f"{stem}.set",
        ROOT / "data/raw" / f"{stem}.fdt",
        ROOT / "data/metadata" / f"{subject}_ses-{session}_task-sleep_acq-scoring1_events.tsv",
    )


def stopped_manifest(manifest: dict, recording: str | None, error: BaseException) -> None:
    manifest["status"] = "stopped_on_error"
    manifest["failure"] = {
        "recording": recording,
        "error_type": type(error).__name__,
        "error": str(error),
        "raw_cleanup_performed": False,
    }
    write_manifest(manifest)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        if manifest.get("status") == "complete" and FINAL_PATH.exists():
            print(f"Dataset already complete: {FINAL_PATH.relative_to(ROOT)}")
            return
        if manifest.get("status") == "stopped_on_error":
            failure = manifest.get("failure", {})
            manifest.setdefault("previous_failures", []).append(failure)
            manifest["failure"] = None
            manifest["status"] = "running"
            write_manifest(manifest)
            print(
                "Continuing from saved night batches after the previous stop at "
                f"{failure.get('recording')}; no completed night batches will be downloaded again.",
                flush=True,
            )
    else:
        if BATCHES.exists() and any(BATCHES.iterdir()):
            raise SystemExit(f"Untracked night batches exist in {BATCHES}; refusing to overwrite them.")
        manifest = {
            "dataset": "OpenNeuro ds005185 (EESM19)",
            "status": "running",
            "subjects": list(SUBJECTS),
            "sessions": list(SESSIONS),
            "session_scope": "PSG nights only; sessions 001–004",
            "ear_signal": "ELE−ERE",
            "epoch_seconds": 30,
            "max_peak_to_peak_epoch_uv": float(CONFIG["max_peak_to_peak_epoch_uv"]),
            "label_source": "Scoring1",
            "excluded_label_codes": {"6": "A", "7": "Artefact", "8": "Unscored"},
            "downloaded_bytes_total": 0,
            "completed_nights": {},
            "failure": None,
        }
        write_manifest(manifest)

    BATCHES.mkdir(parents=True, exist_ok=True)
    mapping_key = "ds005185/task-sleep_events.json"
    current_recording = None
    try:
        print("Reading public object inventory once...", flush=True)
        sizes = inventory()
        planned = [(subject, session) for subject in SUBJECTS for session in SESSIONS]
        needed_keys = [key for subject, session in planned for key in object_keys(subject, session)]
        missing_keys = sorted(set(needed_keys + [mapping_key]) - set(sizes))
        if missing_keys:
            raise FileNotFoundError("Selected PSG files absent from inventory: " + ", ".join(missing_keys))

        mapping_path = ROOT / "data/metadata/task-sleep_events.json"
        if not mapping_path.exists() or mapping_path.stat().st_size != sizes[mapping_key]:
            download(mapping_key, mapping_path, sizes[mapping_key], 0)

        for subject, session in planned:
            recording = f"{subject}:{session}"
            batch_path = BATCHES / f"{subject}_ses-{session}.csv"
            set_path, fdt_path, scoring_path = local_targets(subject, session)
            prior = manifest["completed_nights"].get(recording)
            if prior and prior.get("status") in ("saved", "skipped"):
                if prior.get("status") == "saved" and not batch_path.exists():
                    raise FileNotFoundError(f"Manifest says {recording} is saved, but {batch_path} is missing")
                set_path.unlink(missing_ok=True)
                fdt_path.unlink(missing_ok=True)
                continue

            if recording in SKIPPED_RECORDINGS:
                manifest["completed_nights"][recording] = {
                    "status": "skipped",
                    "reason": SKIPPED_RECORDINGS[recording],
                }
                write_manifest(manifest)
                set_path.unlink(missing_ok=True)
                fdt_path.unlink(missing_ok=True)
                print(f"Skipped {recording}: {SKIPPED_RECORDINGS[recording]}.", flush=True)
                continue

            current_recording = recording
            keys = object_keys(subject, session)
            required_bytes = sum(
                sizes[key] for key, target in zip(keys, (set_path, fdt_path, scoring_path))
                if not target.exists() or target.stat().st_size != sizes[key]
            )
            if required_bytes > int(CONFIG["max_download_bytes"]):
                raise RuntimeError(f"{recording} exceeds the configured per-night download cap")
            downloaded_this_night = 0
            for key, target in zip(keys, (set_path, fdt_path, scoring_path)):
                downloaded_this_night = download(key, target, sizes[key], downloaded_this_night)

            print(f"Extracting features and labels for {recording}", flush=True)
            try:
                features, labels = process_one_night(subject, session)
            except ValueError as error:
                if str(error) != "An ear channel contains no finite samples":
                    raise
                manifest["completed_nights"][recording] = {
                    "status": "skipped",
                    "reason": str(error),
                }
                write_manifest(manifest)
                set_path.unlink(missing_ok=True)
                fdt_path.unlink(missing_ok=True)
                print(f"Skipped {recording}: {error}.", flush=True)
                continue
            if len(features) != len(labels) or not features[KEY_COLUMNS].equals(labels[KEY_COLUMNS]):
                raise ValueError(f"Feature/label epoch keys differ for {recording}")
            if not np.array_equal(features["epoch_index"].to_numpy(), labels["epoch_index"].to_numpy()):
                raise ValueError(f"Feature/label epoch order differs for {recording}")
            feature_values = features[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
            if not np.isfinite(feature_values).all():
                raise ValueError(f"Non-finite feature values remain after measurement for {recording}")

            paired = features.merge(
                labels[KEY_COLUMNS + LABEL_COLUMNS], on=KEY_COLUMNS, how="inner", validate="one_to_one"
            )
            keep = paired["signal_valid"].astype(bool) & paired["stage_code"].isin((1, 2, 3, 4, 5))
            usable = paired.loc[keep].copy()
            if usable.empty:
                missing_epochs = int((paired["nonfinite_sample_fraction"] > 0).sum())
                amplitude_epochs = int(paired["amplitude_outlier"].astype(bool).sum())
                raise ValueError(
                    f"{recording} produced zero usable epochs out of {len(paired)}; "
                    f"epochs with source NaNs={missing_epochs}, amplitude outliers={amplitude_epochs}. "
                    "Inspect MNE scaling and the peak-to-peak quality limit before continuing."
                )
            temporary_batch = batch_path.with_suffix(".csv.tmp")
            usable.to_csv(temporary_batch, index=False)
            temporary_batch.replace(batch_path)

            label_counts = labels["stage_name"].value_counts().to_dict()
            manifest["completed_nights"][recording] = {
                "status": "saved",
                "source_epochs": int(len(paired)),
                "usable_epochs_saved": int(len(usable)),
                "unusable_signal_epochs_dropped": int((~paired["signal_valid"].astype(bool)).sum()),
                "source_nan_epochs": int((paired["nonfinite_sample_fraction"] > 0).sum()),
                "amplitude_outlier_epochs": int(paired["amplitude_outlier"].astype(bool).sum()),
                "label_epochs_dropped": int((~paired["stage_code"].isin((1, 2, 3, 4, 5))).sum()),
                "label_counts": {str(key): int(value) for key, value in label_counts.items()},
                "downloaded_bytes": int(downloaded_this_night),
                "batch_file": str(batch_path.relative_to(ROOT)),
            }
            manifest["downloaded_bytes_total"] += int(downloaded_this_night)
            write_manifest(manifest)

            # Remove the large raw pair only after this night's feature/label batch is durable.
            set_path.unlink(missing_ok=True)
            fdt_path.unlink(missing_ok=True)
            print(f"Saved {len(usable)} usable epochs for {recording}; removed raw EEG pair.", flush=True)

        unfinished = [
            f"{subject}:{session}" for subject, session in planned
            if manifest["completed_nights"].get(f"{subject}:{session}", {}).get("status") not in ("saved", "skipped")
        ]
        if unfinished:
            raise FileNotFoundError("Night batches not completed after processing: " + ", ".join(unfinished))
        batch_paths = [
            BATCHES / f"{subject}_ses-{session}.csv" for subject, session in planned
            if manifest["completed_nights"][f"{subject}:{session}"].get("status") == "saved"
        ]
        tables = [pd.read_csv(path) for path in batch_paths]
        combined = pd.concat(tables, ignore_index=True)
        if combined.empty:
            raise ValueError("No usable labeled epochs remain in the selected cohort")
        temporary_final = FINAL_PATH.with_suffix(".csv.tmp")
        combined.to_csv(temporary_final, index=False)
        temporary_final.replace(FINAL_PATH)
        manifest["status"] = "complete"
        manifest["final_rows"] = int(len(combined))
        manifest["final_file"] = str(FINAL_PATH.relative_to(ROOT))
        write_manifest(manifest)
        print(f"Complete: {len(combined):,} rows in {FINAL_PATH.relative_to(ROOT)}", flush=True)
    except BaseException as error:
        stopped_manifest(manifest, current_recording, error)
        print(f"STOPPED on {current_recording or 'dataset preflight'}: {error}", flush=True)
        print("Any raw files for the failed night were retained for diagnosis; no automatic retry was attempted.", flush=True)
        raise


if __name__ == "__main__":
    main()
