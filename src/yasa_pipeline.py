"""Summarize YASA sleep staging from a local ds005185 in-ear EEG recording."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yasa

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results" / "yasa"
STAGES = (1, 2, 3, 4, 5)
STAGE_NAMES = {1: "Wake", 2: "REM", 3: "N1", 4: "N2", 5: "N3"}
SIGNALS = {
    "in_ear": {"name": "In-ear EEG (ELE−ERE)", "channels": ("ELE", "ERE")},
    "scalp_psg": {"name": "Scalp PSG (C3−M2)", "channels": ("C3", "M2")},
}


def stage_mapping(path: Path) -> dict[int, str]:
    doc = json.loads(path.read_text())
    return {int(k.removeprefix("s")): str(v) for k, v in doc["Scoring1"]["levels"].items()}


def score(y_true: np.ndarray, predictions: list[str], mapping: dict[int, str]) -> dict:
    code_by_name = {name.upper(): code for code, name in mapping.items()}
    aliases = {"W": "WAKE", "R": "REM"}
    predicted = np.array([
        code_by_name.get(aliases.get(str(label).upper(), str(label).upper()), -1)
        for label in predictions
    ])
    valid = np.isin(y_true, STAGES) & np.isin(predicted, STAGES)
    if not valid.any():
        raise RuntimeError("No scored sleep epochs available for statistics")
    errors = y_true[valid] != predicted[valid]
    by_stage = {}
    for code in STAGES:
        mask = valid & (y_true == code)
        count = int(mask.sum())
        error_pct = float(100 * np.mean(y_true[mask] != predicted[mask])) if count else None
        by_stage[mapping.get(code, STAGE_NAMES[code])] = {
            "epochs": count,
            "errors": int(np.sum(y_true[mask] != predicted[mask])),
            "error_percent": error_pct,
        }
    ranked = [(name, stats["error_percent"]) for name, stats in by_stage.items()
              if stats["error_percent"] is not None]
    weakest_stage, weakest_error = max(ranked, key=lambda item: item[1])
    return {
        "epochs_compared": int(valid.sum()),
        "errors": int(errors.sum()),
        "error_percent": float(100 * errors.mean()),
        "accuracy_percent": float(100 * (1 - errors.mean())),
        "weakest_stage": weakest_stage,
        "weakest_stage_error_percent": weakest_error,
        "by_stage": by_stage,
    }


def predict_stages(signal: np.ndarray, sfreq: float, channel: str, label_count: int) -> np.ndarray:
    model_input = mne.io.RawArray(
        signal[np.newaxis, :], mne.create_info([channel], sfreq, ["eeg"]), verbose="ERROR"
    )
    predictions = yasa.SleepStaging(
        model_input, eeg_name=channel, eog_name=None, emg_name=None
    ).predict().hypno.to_numpy(dtype=str)
    if len(predictions) != label_count:
        raise ValueError(f"YASA returned {len(predictions)} {channel} epochs for {label_count} labels")
    return predictions


def process(recording: str) -> dict:
    try:
        subject, session = recording.split(":", 1)
        if not re.fullmatch(r"sub-\d+", subject) or not re.fullmatch(r"\d{3}", session):
            raise ValueError
    except ValueError as exc:
        raise SystemExit("Use --recording SUBJECT:SESSION, e.g. sub-016:001") from exc

    stem = f"{subject}_ses-{session}_task-sleep_acq-PSG_eeg"
    raw_path = RAW / (stem + ".set")
    fdt_path = RAW / (stem + ".fdt")
    scoring_path = DATA / "metadata" / f"{subject}_ses-{session}_task-sleep_acq-scoring1_events.tsv"
    mapping_path = DATA / "metadata" / "task-sleep_events.json"
    needed = (raw_path, fdt_path, scoring_path, mapping_path)
    absent = [path for path in needed if not path.exists()]
    if absent:
        raise FileNotFoundError(
            "Input files are missing. Run `python3 scripts/fetch_data.py --recording "
            f"{recording}` first. Missing: " + ", ".join(str(path.relative_to(ROOT)) for path in absent)
        )

    mapping = stage_mapping(mapping_path)
    table = pd.read_csv(scoring_path, sep="\t")
    labels = table["Scoring1"].astype(int).to_numpy()
    onsets = table["onset"].astype(float).to_numpy()
    durations = table["duration"].astype(float).to_numpy()
    epoch_seconds = int(CONFIG["epoch_seconds"])
    if not (len(labels) == len(onsets) == len(durations)) or not np.allclose(durations, epoch_seconds):
        raise ValueError("Sleep labels do not have the expected epoch timing")

    raw = mne.io.read_raw_eeglab(str(raw_path), preload=False, verbose="ERROR")
    needed_channels = ["ELE", "ERE", "C3", "M2"]
    missing_channels = sorted(set(needed_channels) - set(raw.ch_names))
    if missing_channels:
        raise ValueError("Missing EEG channels: " + ", ".join(missing_channels))
    sfreq_original = float(raw.info["sfreq"])
    raw.pick(needed_channels)
    raw.load_data()
    raw.resample(float(CONFIG["sampling_rate"]), npad="auto", verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    n_samples = int(round(epoch_seconds * sfreq))
    starts = np.rint(onsets * sfreq).astype(int)
    samples = raw.get_data()
    if not len(starts) or np.any(starts < 0) or np.any(starts + n_samples > samples.shape[1]):
        raise ValueError("Scored epochs fall outside the EEG recording")

    # Keep the PSG baseline and ear derivation aligned to the same Scoring1 epochs.
    names = {name: i for i, name in enumerate(raw.ch_names)}
    start, end = int(starts[0]), int(starts[-1] + n_samples)
    if end - start != len(labels) * n_samples:
        raise ValueError("EEG and scored epochs do not align on a continuous 30-second grid")
    aligned = samples[:, start:end]
    predictions = {}
    summaries = {}
    for key, spec in SIGNALS.items():
        positive, negative = spec["channels"]
        signal = aligned[names[positive]] - aligned[names[negative]]
        predictions[key] = predict_stages(signal, sfreq, key, len(labels))
        summaries[key] = score(labels, predictions[key].tolist(), mapping)

    result = {
        "recording": recording,
        "sampling_rate_hz": sfreq,
        "original_sampling_rate_hz": sfreq_original,
        "signals": {
            key: {"name": SIGNALS[key]["name"], "summary": summaries[key],
                  "yasa_stage_counts": {stage: int(np.sum(predictions[key] == stage))
                                        for stage in sorted(set(predictions[key]))}}
            for key in SIGNALS
        },
        "ear_minus_scalp_error_percentage_points": round(
            summaries["in_ear"]["error_percent"] - summaries["scalp_psg"]["error_percent"], 2
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    result_stem = f"{subject}_ses-{session}_comparison"
    (RESULTS / f"{result_stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"# YASA signal comparison: {recording}", "",
             f"Compared on the same {summaries['in_ear']['epochs_compared']} valid Scoring1 epochs.", "",
             "## Overall", "", "| Signal | Error | Accuracy | Weakest stage | Stage error |", "|---|---:|---:|---|---:|"]
    for key, spec in SIGNALS.items():
        summary = summaries[key]
        lines.append(f"| {spec['name']} | {summary['error_percent']:.1f}% | "
                     f"{summary['accuracy_percent']:.1f}% | {summary['weakest_stage']} | "
                     f"{summary['weakest_stage_error_percent']:.1f}% |")
    gap = result["ear_minus_scalp_error_percentage_points"]
    lines += ["", f"**In-ear error minus scalp PSG error:** {gap:+.2f} percentage points.", "",
              "## Error by stage", "", "| Stage | Labeled epochs | In-ear error | Scalp PSG error | Ear minus scalp |",
              "|---|---:|---:|---:|---:|"]
    for stage, ear_stats in summaries["in_ear"]["by_stage"].items():
        scalp_stats = summaries["scalp_psg"]["by_stage"][stage]
        ear_error, scalp_error = ear_stats["error_percent"], scalp_stats["error_percent"]
        ear_text = "—" if ear_error is None else f"{ear_error:.1f}%"
        scalp_text = "—" if scalp_error is None else f"{scalp_error:.1f}%"
        gap_text = "—" if ear_error is None or scalp_error is None else f"{ear_error - scalp_error:+.1f} pp"
        lines.append(f"| {stage} | {ear_stats['epochs']} | {ear_text} | {scalp_text} | {gap_text} |")
    for key, spec in SIGNALS.items():
        counts = result["signals"][key]["yasa_stage_counts"]
        lines += ["", f"YASA predictions for {spec['name']}: " + ", ".join(
            f"{stage} {count}" for stage, count in counts.items()) + "."]
    (RESULTS / f"{result_stem}.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare YASA staging on local in-ear and scalp PSG EEG")
    parser.add_argument("--recording", default="sub-016:001", metavar="SUBJECT:SESSION")
    args = parser.parse_args()
    print(json.dumps(process(args.recording), indent=2))


if __name__ == "__main__":
    main()
