"""Download-once, process-once ds005185 PSG pipeline for YASA smoke/cohort runs."""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yasa
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
META = DATA / "metadata"
RESULTS = ROOT / "results"
BUCKET = "s3://openneuro.org/"
EPOCH_SEC = int(CONFIG["epoch_seconds"])
BAD_CODES = (6, 7, 8)
STAGE_ORDER = (1, 2, 3, 4, 5)


def json_safe(value):
    """Convert NumPy/Pandas values and non-string mapping keys for JSON."""
    if isinstance(value, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def parse_label_mapping(json_path: Path):
    """Parse code levels from the BIDS JSON that describes Scoring1/Scoring2."""
    doc = json.loads(json_path.read_text())
    mapping = {int(k.removeprefix("s")): str(v) for k, v in doc["Scoring1"]["levels"].items()}
    return mapping, doc


def inventory_from_s3(listing: list[str]):
    sizes: dict[str, int] = {}
    pattern = re.compile(r"\s*\d{4}-\d\d-\d\d\s+\d\d:\d\d:\d\d\s+(\d+)\s+(.+)$")
    for line in listing:
        match = pattern.match(line)
        if match:
            sizes[match.group(2)] = int(match.group(1))
    file_re = re.compile(r"ds005185/(sub-\d+)/ses-(\d+)/eeg/\1_ses-\2_task-sleep_acq-PSG_eeg\.(set|fdt)$")
    pairs = {}
    for key, size in sizes.items():
        m = file_re.fullmatch(key)
        if m and 1 <= int(m.group(2)) <= 4:
            pairs.setdefault((m.group(1), m.group(2)), {})[m.group(3)] = (key, size)
    recordings = []
    for (sub, ses), pair in pairs.items():
        if set(pair) == {"set", "fdt"}:
            recordings.append({"subject": sub, "session": ses, "set": pair["set"],
                               "fdt": pair["fdt"], "raw_bytes": sum(p[1] for p in pair.values())})
    return sorted(recordings, key=lambda r: (r["subject"], int(r["session"]))), sizes


def s3_listing():
    p = subprocess.run(["aws", "s3", "ls", "--no-sign-request", "--recursive", BUCKET + "ds005185/"],
                       capture_output=True, text=True, check=True)
    return p.stdout.splitlines()


def disk_bytes():
    return sum(p.stat().st_size for base in (DATA, RESULTS) if base.exists()
               for p in base.rglob("*") if p.is_file())


def download(key: str, target: Path, expected: int, meter: dict):
    """Check both hard caps before every object transfer and verify its exact size."""
    if target.exists() and target.stat().st_size == expected:
        return 0
    if int(meter["downloaded_bytes"]) + expected > int(CONFIG["max_download_bytes"]):
        raise RuntimeError(f"18 GB download cap reached before {key}")
    if disk_bytes() + expected > int(CONFIG["max_disk_bytes"]):
        raise RuntimeError(f"20 GB disk cap reached before {key}")
    target.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["aws", "s3", "cp", "--no-sign-request", BUCKET + key, str(target)],
                       capture_output=True, text=True)
    actual = target.stat().st_size if target.exists() else 0
    meter["downloaded_bytes"] += actual
    meter["peak_disk_bytes"] = max(meter["peak_disk_bytes"], disk_bytes())
    if p.returncode or actual != expected:
        target.unlink(missing_ok=True)
        raise IOError(f"download incomplete {key}: expected {expected}, received {actual}; {p.stderr[-500:]}")
    return actual


def recording_keys(rec):
    prefix = f"ds005185/{rec['subject']}/ses-{rec['session']}/eeg/{rec['subject']}_ses-{rec['session']}_task-sleep_acq-"
    return {"scoring1": prefix + "scoring1_events.tsv",
            "scoring2": prefix + "scoring2_events.tsv"}


def qc_channels(raw_uv: np.ndarray, names: list[str], sfreq: float):
    """Flatness, extreme amplitude, and pre-notch 50 Hz line-noise ratio."""
    report = {}
    n = min(raw_uv.shape[1], int(sfreq * 4))
    f = np.fft.rfftfreq(n, 1 / sfreq)
    for name, signal in zip(names, raw_uv):
        spectrum = np.abs(np.fft.rfft(signal[:n])) ** 2
        line = float(spectrum[np.abs(f - float(CONFIG["line_frequency_hz"])) <= 1].sum())
        total = float(spectrum[(f >= 1) & (f <= min(40, sfreq / 2))].sum())
        sd, peak = float(np.std(signal)), float(np.max(np.abs(signal)))
        ratio = line / max(total, 1e-30)
        reasons = []
        if sd < float(CONFIG["flat_std_uv"]): reasons.append("flat")
        if peak > float(CONFIG["extreme_abs_uv"]): reasons.append("extreme_amplitude")
        if ratio > float(CONFIG["line_noise_ratio_limit"]): reasons.append("line_noise")
        report[name] = {"std_uv": sd, "peak_abs_uv": peak, "line_noise_ratio": ratio,
                        "usable": not reasons, "reasons": reasons}
    return report


def yasa_labels(raw: mne.io.BaseRaw, channel: str, n_labels: int):
    hypnogram = yasa.SleepStaging(raw, eeg_name=channel, eog_name=None, emg_name=None).predict()
    # YASA returns a Hypnogram; read its stored vector directly (avoids slow slicing).
    labels = hypnogram.hypno.to_numpy(dtype=str)
    validate_prediction_length(labels, n_labels, channel)
    return labels


def validate_prediction_length(predictions, label_count: int, channel="prediction"):
    assert len(predictions) == int(label_count), f"{channel}: {len(predictions)} predictions != {label_count} labels"


def stage_codes(predictions: np.ndarray, mapping: dict[int, str]):
    label_codes = {name.upper(): int(code) for code, name in mapping.items()}
    aliases = {"W": "WAKE", "WAKE": "WAKE", "R": "REM"}
    return np.asarray([label_codes.get(aliases.get(str(x).upper(), str(x).upper()), -1)
                       for x in predictions], dtype=int)


def metrics(y_true: np.ndarray, pred_codes: np.ndarray, mapping: dict[int, str]):
    valid = ~np.isin(y_true, BAD_CODES) & np.isin(y_true, STAGE_ORDER) & np.isin(pred_codes, STAGE_ORDER)
    yt, yp = y_true[valid], pred_codes[valid]
    names = [mapping[i] for i in STAGE_ORDER]
    return {"n_evaluated": int(valid.sum()),
            "accuracy": float(accuracy_score(yt, yp)),
            "cohen_kappa": float(cohen_kappa_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, labels=STAGE_ORDER, average="macro", zero_division=0)),
            "per_class_f1": {mapping[i]: float(v) for i, v in zip(STAGE_ORDER,
                              f1_score(yt, yp, labels=STAGE_ORDER, average=None, zero_division=0))},
            "confusion_matrix": confusion_matrix(yt, yp, labels=STAGE_ORDER).tolist(),
            "confusion_labels": names}


def class_counts(labels: np.ndarray, mapping: dict[int, str]):
    counts = Counter(int(x) for x in labels)
    return {mapping.get(k, f"code_{k}"): int(counts.get(k, 0)) for k in sorted(mapping)}


def process_recording(rec, sizes, meter, *, keep_raw_on_error=True):
    subject, session = rec["subject"], rec["session"]
    stem = f"{subject}_ses-{session}_task-sleep_acq-PSG_eeg"
    set_path, fdt_path = RAW / (stem + ".set"), RAW / (stem + ".fdt")
    RAW.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    score_keys = recording_keys(rec)
    # Common small metadata plus recording-specific event TSVs.
    common = {"dataset_description.json": "ds005185/dataset_description.json",
              "participants.tsv": "ds005185/participants.tsv",
              "task-sleep_events.json": "ds005185/task-sleep_events.json",
              "task-sleep_acq-scoring_events.json": "ds005185/task-sleep_acq-scoring_events.json",
              "acq-PSG_channels.tsv": "ds005185/acq-PSG_channels.tsv"}
    sidecar_items = {**common, **{Path(k).name: k for k in score_keys.values()}}
    for key in sidecar_items.values():
        size = sizes.get(key)
        if size is None:
            raise FileNotFoundError(f"sidecar absent from pre-download inventory: {key}")
        print(f"SIDECAR INVENTORY {key} {size} bytes", flush=True)
    for name, key in sidecar_items.items():
        size = sizes[key]
        download(key, META / name, size, meter)
    map_path = META / "task-sleep_events.json"
    mapping, task_doc = parse_label_mapping(map_path)
    print(f"Label mapping source: {map_path.relative_to(ROOT)}; mapping: {mapping}", flush=True)
    expected_map = {1: "Wake", 2: "REM", 3: "N1", 4: "N2", 5: "N3", 6: "A", 7: "Artefact", 8: "Unscored"}
    if mapping != expected_map:
        raise RuntimeError(f"DEVASTATING: unexpected runtime label mapping: {mapping}")
    scoring_json_path = META / "task-sleep_acq-scoring_events.json"
    acquisition_doc = json.loads(scoring_json_path.read_text())
    scoring2_description = task_doc.get("Scoring2", {}).get("description")
    scoring2_level_map = task_doc.get("Scoring2", {}).get("levels", {})
    # This separate acquisition JSON describes diary/recording events, not Scoring2.
    scoring2_in_acq_json = acquisition_doc.get("Scoring2")

    try:
        download(rec["set"][0], set_path, rec["set"][1], meter)
        download(rec["fdt"][0], fdt_path, rec["fdt"][1], meter)
    except BaseException:
        # A pair is useful only when both files match the preflight inventory.
        set_path.unlink(missing_ok=True)
        fdt_path.unlink(missing_ok=True)
        raise
    try:
        s1 = pd.read_csv(META / Path(score_keys["scoring1"]).name, sep="\t")
        s2 = pd.read_csv(META / Path(score_keys["scoring2"]).name, sep="\t")
        if "Scoring1" not in s1 or "Scoring2" not in s2 or len(s1) != len(s2):
            raise AssertionError("Scoring1/Scoring2 column or row-count mismatch")
        y = s1["Scoring1"].astype(int).to_numpy()
        y2 = s2["Scoring2"].astype(int).to_numpy()
        onset = s1["onset"].astype(float).to_numpy()
        duration = s1["duration"].astype(float).to_numpy()
        assert len(onset) == len(y), "scoring onset/label count mismatch"
        assert np.allclose(duration, EPOCH_SEC), f"unexpected durations: {np.unique(duration)}"
        assert len(onset) == 0 or np.allclose(np.diff(onset), EPOCH_SEC, atol=0.02), "scoring epochs are not regularly spaced"

        raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose="ERROR")
        required = list(CONFIG["ear_channels"]) + ["C3", "M2"]
        missing = sorted(set(required) - set(raw.ch_names))
        if missing:
            raise RuntimeError(f"missing channels: {missing}")
        raw.pick(required)
        raw.load_data()
        sfreq_orig = float(raw.info["sfreq"])
        qc = qc_channels(raw.get_data(picks=CONFIG["ear_channels"]) * 1e6,
                         CONFIG["ear_channels"], sfreq_orig)
        if len(CONFIG["ear_channels"]) != 12 or raw.n_times == 0:
            raise RuntimeError("DEVASTATING: ear channels missing or signal empty")
        if all(not q["usable"] or q["std_uv"] < float(CONFIG["flat_std_uv"])
               for q in qc.values()):
            raise RuntimeError("DEVASTATING: all ear channels unusable/flat")
        raw.filter(*map(float, CONFIG["filter_hz"]), verbose="ERROR")
        raw.notch_filter(float(CONFIG["line_frequency_hz"]), verbose="ERROR")
        raw.resample(float(CONFIG["sampling_rate"]), npad="auto", verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        n_samp = int(round(EPOCH_SEC * sfreq))
        starts = np.rint(onset * sfreq).astype(int)
        data_uv = raw.get_data(picks=required) * 1e6
        if len(y) != len(s1) or any(st < 0 or st + n_samp > data_uv.shape[1] for st in starts):
            raise AssertionError("DEVASTATING: label count/alignment exceeds signal bounds")
        epochs = np.stack([data_uv[:, st:st + n_samp] for st in starts])
        assert epochs.shape == (len(y), len(required), n_samp), "epoch count/shape mismatch"
        ear_indices = [required.index(c) for c in CONFIG["ear_channels"]]
        ear = epochs[:, ear_indices, :]
        # A priori ear comparison set: ELE−ERE plus each ear's contacts referenced
        # to ELE/ERE. No test-fold selection is performed during this smoke run.
        ci = {name: i for i, name in enumerate(CONFIG["ear_channels"])}
        common_ref = np.stack([ear[:, ci[c], :] - ear[:, ci["ELE" if c.startswith("EL") else "ERE"], :]
                               for c in CONFIG["ear_channels"] if c not in ("ELE", "ERE")], axis=1)
        bipolar = (ear[:, ci["ELE"], :] - ear[:, ci["ERE"], :])
        assert bipolar.shape[0] == len(y) and bipolar.size > 0
        assert float(np.std(bipolar)) > 0, "DEVASTATING: ear derivation is flat"

        first_sample = int(starts[0])
        end_sample = int(starts[-1] + n_samp)
        aligned_raw = raw.copy().crop(tmin=first_sample / sfreq,
                                      tmax=end_sample / sfreq, include_tmax=False)
        if aligned_raw.n_times != len(y) * n_samp:
            aligned_raw = mne.io.RawArray(raw.get_data(start=first_sample, stop=end_sample),
                                           raw.info.copy(), verbose="ERROR")
        assert aligned_raw.n_times == len(y) * n_samp, "DEVASTATING: YASA crop is not exact"
        scalp_v = aligned_raw.get_data(picks=["C3"])[0] - aligned_raw.get_data(picks=["M2"])[0]
        ear_v = bipolar.reshape(1, -1) * 1e-6
        assert ear_v.shape[1] == len(y) * n_samp
        scalp_raw = mne.io.RawArray(scalp_v[None, :], mne.create_info(["C3_M2"], sfreq, ["eeg"]), verbose="ERROR")
        ear_raw = mne.io.RawArray(ear_v, mne.create_info(["ELE_ERE"], sfreq, ["eeg"]), verbose="ERROR")
        pred_scalp = yasa_labels(scalp_raw, "C3_M2", len(y))
        pred_ear = yasa_labels(ear_raw, "ELE_ERE", len(y))
        assert len(pred_scalp) == len(pred_ear) == len(y), "DEVASTATING: per-recording predictions != labels"
        scalp_metrics = metrics(y, stage_codes(pred_scalp, mapping), mapping)
        ear_metrics = metrics(y, stage_codes(pred_ear, mapping), mapping)
        if scalp_metrics["cohen_kappa"] < 0.5:
            raise RuntimeError(f"DEVASTATING: YASA-scalp kappa {scalp_metrics['cohen_kappa']:.3f} < 0.5")
        valid12 = ~np.isin(y, BAD_CODES) & ~np.isin(y2, BAD_CODES)
        irater = {"n_epochs": int(valid12.sum()),
                  "accuracy": float(accuracy_score(y[valid12], y2[valid12])) if valid12.any() else None,
                  "cohen_kappa": float(cohen_kappa_score(y[valid12], y2[valid12])) if valid12.any() else None}
        excluded = {mapping[c]: int(np.sum(y == c)) for c in BAD_CODES}
        result = {
            "subject": subject, "session": session, "n_epochs": len(y),
            "class_counts": class_counts(y, mapping), "label_mapping": mapping,
            "label_mapping_source": str(map_path.relative_to(ROOT)),
            "scoring2_description": scoring2_description,
            "scoring2_description_source": str(map_path.relative_to(ROOT)),
            "scoring2_present_in_task_sleep_acq_scoring_events_json": scoring2_in_acq_json is not None,
            "task_sleep_acq_scoring_events_json_fields": sorted(acquisition_doc),
            "scoring2_code_mapping": scoring2_level_map,
            "scoring2_independence_verified": False,
            "scoring1_vs_scoring2": irater,
            "excluded_label_counts": excluded,
            "yasa_scalp": scalp_metrics, "yasa_ear": ear_metrics,
            "yasa_prediction_count": {"scalp": len(pred_scalp), "ear": len(pred_ear)},
            "signal": {"original_sfreq_hz": sfreq_orig, "sfreq_hz": sfreq,
                       "raw_samples": int(raw.n_times), "scored_onset_first_s": float(onset[0]),
                       "aligned_sample_count": int(aligned_raw.n_times), "ear_epochs": int(ear.shape[0]),
                       "ear_channels": list(CONFIG["ear_channels"]),
                       "ear_derivations": ["ELE-ERE", "left contacts vs ELE", "right contacts vs ERE"]},
            "channel_quality": qc,
            "unusable_ear_channels": [k for k, v in qc.items() if not v["usable"]],
            "raw_pair_bytes": int(rec["raw_bytes"]),
            "downloaded_bytes_so_far": int(meter["downloaded_bytes"]),
        }
        cache_path = CACHE / f"{subject}_ses-{session}_epochs.npz"
        np.savez_compressed(cache_path, labels=y, labels_scoring2=y2,
                            valid_mask=~np.isin(y, BAD_CODES), onset_seconds=onset,
                            ear_raw_uv=ear.astype(np.float32),
                            ear_common_ref_uv=common_ref.astype(np.float32),
                            ear_bipolar_ele_ere_uv=bipolar[:, None, :].astype(np.float32),
                            scalp_c3_m2_uv=(epochs[:, -2, :] - epochs[:, -1, :]).astype(np.float32),
                            yasa_scalp=pred_scalp.astype("U8"), yasa_ear=pred_ear.astype("U8"),
                            subject=np.asarray(subject), session=np.asarray(session))
        result["cache_bytes"] = int(cache_path.stat().st_size)
        meter["peak_disk_bytes"] = max(meter["peak_disk_bytes"], disk_bytes())
        if meter["peak_disk_bytes"] > int(CONFIG["max_disk_bytes"]):
            raise RuntimeError("DEVASTATING: generated project data approached/exceeded 20 GB")
        log_path = RESULTS / f"recording_{subject}_ses-{session}.json"
        write_json(log_path, result)  # durable per-recording log before raw deletion
        print(f"DONE {subject} ses-{session} epochs={len(y)} scalp_kappa={scalp_metrics['cohen_kappa']:.3f} "
              f"ear_kappa={ear_metrics['cohen_kappa']:.3f} bytes={meter['downloaded_bytes']}", flush=True)
        return result
    except BaseException:
        if not keep_raw_on_error:
            set_path.unlink(missing_ok=True)
            fdt_path.unlink(missing_ok=True)
        raise
    finally:
        # Successful raw deletion happens only after metrics and per-recording log.
        if (RESULTS / f"recording_{subject}_ses-{session}.json").exists():
            set_path.unlink(missing_ok=True)
            fdt_path.unlink(missing_ok=True)


def candidates_and_sizes():
    lines = s3_listing()
    return inventory_from_s3(lines)


def select_cohort(recordings):
    ses1 = [r for r in recordings if r["session"] == "001"]
    if not any(r["subject"] == "sub-016" for r in ses1):
        raise RuntimeError("sub-016 ses-001 is missing from PSG inventory")
    ordered = [next(r for r in ses1 if r["subject"] == "sub-016")]
    ordered += sorted((r for r in ses1 if r["subject"] != "sub-016"), key=lambda x: x["raw_bytes"])
    selected, used = [], 0
    for r in ordered:
        if len(selected) >= int(CONFIG["target_subjects"]): break
        if used + r["raw_bytes"] <= int(CONFIG["max_download_bytes"]):
            selected.append(r); used += r["raw_bytes"]
    if len(selected) < int(CONFIG["minimum_subjects"]):
        raise RuntimeError(f"only {len(selected)} distinct ses-001 subjects fit the configured cap")
    return selected


def download_manifest_text(recordings, sizes):
    """Create the compact manifest from pre-download inventory, without data transfer."""
    result = []
    common = ["ds005185/dataset_description.json", "ds005185/participants.tsv",
              "ds005185/task-sleep_events.json", "ds005185/task-sleep_acq-scoring_events.json",
              "ds005185/acq-PSG_channels.tsv"]
    for key in common:
        result.append({"subject": "ALL", "session": "ALL", "path": key,
                       "expected_size_bytes": sizes.get(key, ""),
                       "status": "shared_sidecar" if key in sizes else "missing"})
    for r in recordings:
        result.append({"subject": r["subject"], "session": r["session"],
                       "path": r["set"][0].removeprefix("ds005185/"),
                       "expected_size_bytes": r["set"][1],
                       "status": "confirmed_readable" if (r["subject"], r["session"]) == ("sub-016", "001") else "pending_selection_and_readability"})
        result.append({"subject": r["subject"], "session": r["session"],
                       "path": r["fdt"][0].removeprefix("ds005185/"),
                       "expected_size_bytes": r["fdt"][1],
                       "status": "confirmed_readable" if (r["subject"], r["session"]) == ("sub-016", "001") else "pending_selection_and_readability"})
        for kind, key in recording_keys(r).items():
            result.append({"subject": r["subject"], "session": r["session"],
                           "path": key.removeprefix("ds005185/"),
                           "expected_size_bytes": sizes.get(key, ""),
                           "status": "confirmed_readable" if (r["subject"], r["session"]) == ("sub-016", "001") else "pending_selection_and_readability"})
    return result


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subject", "session", "path", "expected_size_bytes", "status"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def run_recordings(recordings, sizes, *, smoke=False, resume=False, force=False, all_recordings=None):
    prior_progress = RESULTS / "run_progress.json"
    previous = json.loads(prior_progress.read_text()) if resume and prior_progress.exists() else {}
    meter = {"downloaded_bytes": int(previous.get("downloaded_bytes", 0)),
             "peak_disk_bytes": max(disk_bytes(), int(previous.get("peak_disk_bytes", 0)))}
    all_recordings = all_recordings or recordings
    completed = []
    for rec in recordings:
        cache_path = CACHE / f"{rec['subject']}_ses-{rec['session']}_epochs.npz"
        log_path = RESULTS / f"recording_{rec['subject']}_ses-{rec['session']}.json"
        if resume and not force and cache_path.exists() and log_path.exists():
            prior = json.loads(log_path.read_text())
            completed.append(prior)
            print(f"RESUME {rec['subject']} ses-{rec['session']} cache already complete; "
                  f"epochs={prior['n_epochs']} scalp_kappa={prior['yasa_scalp']['cohen_kappa']:.3f} "
                  f"ear_kappa={prior['yasa_ear']['cohen_kappa']:.3f} bytes={meter['downloaded_bytes']}", flush=True)
            continue
        if force:
            cache_path.unlink(missing_ok=True); log_path.unlink(missing_ok=True)
        try:
            result = process_recording(rec, sizes, meter, keep_raw_on_error=smoke)
            completed.append(result)
        except Exception as e:
            fail = {"subject": rec["subject"], "session": rec["session"], "error": str(e),
                    "downloaded_bytes": meter["downloaded_bytes"], "peak_disk_bytes": meter["peak_disk_bytes"]}
            write_json(RESULTS / f"problem_{rec['subject']}_ses-{rec['session']}.json", fail)
            if smoke:
                top = "# HANDOFF — STOPPED ON SMOKE-TEST FAILURE\n\n"
                top += f"Devastating-error check failed for {rec['subject']} ses-{rec['session']}: {e}\n\n"
                top += "Evidence is saved in `results/problem_" + rec["subject"] + "_ses-" + rec["session"] + ".json`. "
                top += "Data was retained for debugging. Do not treat the pipeline as handoff-ready.\n"
                (ROOT / "HANDOFF.md").write_text(top)
                raise
            print(f"SKIP {rec['subject']} ses-{rec['session']}: {e}", flush=True)
            # Prefer ses-001; try a later PSG night only when ses-001 cannot be read.
            if rec["session"] == "001":
                alternatives = sorted((r for r in all_recordings
                                       if r["subject"] == rec["subject"] and int(r["session"]) > 1),
                                      key=lambda r: int(r["session"]))
                for fallback in alternatives:
                    try:
                        print(f"FALLBACK {fallback['subject']} ses-{fallback['session']}", flush=True)
                        result = process_recording(fallback, sizes, meter, keep_raw_on_error=False)
                        completed.append(result)
                        break
                    except Exception as fallback_error:
                        write_json(RESULTS / f"problem_{fallback['subject']}_ses-{fallback['session']}.json",
                                   {"subject": fallback["subject"], "session": fallback["session"],
                                    "error": str(fallback_error), "downloaded_bytes": meter["downloaded_bytes"]})
        # Write cohort summary after every recording, not only at the end.
        write_json(RESULTS / "run_progress.json", {"completed": completed, "downloaded_bytes": meter["downloaded_bytes"],
                                                    "peak_disk_bytes": meter["peak_disk_bytes"]})
    return completed, meter


def write_smoke_summary(result, meter):
    write_json(RESULTS / "smoke_test_sub-016.json", result)
    lines = ["# sub-016 ses-001 smoke test", "",
             f"- Epochs: {result['n_epochs']} (Scoring1 onsets; first onset {result['signal']['scored_onset_first_s']:.3f} s)",
             f"- Mapping source: `{result['label_mapping_source']}`",
             f"- Mapping: `{result['label_mapping']}`",
             f"- Total bytes downloaded this smoke run: {meter['downloaded_bytes']:,}",
             f"- Raw pair size: {result['raw_pair_bytes']:,} bytes",
             f"- Scoring2: {result['scoring2_description']}; descriptor present in requested acquisition JSON: {result['scoring2_present_in_task_sleep_acq_scoring_events_json']}",
             f"- Scoring1 vs Scoring2: accuracy {result['scoring1_vs_scoring2']['accuracy']:.3f}, kappa {result['scoring1_vs_scoring2']['cohen_kappa']:.3f} over {result['scoring1_vs_scoring2']['n_epochs']} jointly valid epochs",
             "", "## Class counts", "", "| Stage/code | Epochs |", "|---|---:|"]
    lines.extend(f"| {label} | {count} |" for label, count in result["class_counts"].items())
    for key, title in (("yasa_scalp", "YASA scalp C3−M2"), ("yasa_ear", "YASA ear ELE−ERE")):
        m = result[key]
        lines += ["", f"## {title}", "",
                  f"Accuracy {m['accuracy']:.3f}; Cohen’s kappa {m['cohen_kappa']:.3f}; macro F1 {m['macro_f1']:.3f}.",
                  "", f"Confusion matrix, labels {', '.join(m['confusion_labels'])}:", "", "```text",
                  "\n".join(" ".join(str(int(x)) for x in row) for row in m["confusion_matrix"]), "```", "",
                  "Per-class F1: " + "; ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items())]
    lines += ["", "## Ear channel quality", "", "| Channel | SD (µV) | Peak (µV) | 50 Hz ratio | Usable | Reason |", "|---|---:|---:|---:|---|---|"]
    for channel, q in result["channel_quality"].items():
        lines.append(f"| {channel} | {q['std_uv']:.3f} | {q['peak_abs_uv']:.3f} | {q['line_noise_ratio']:.4f} | {q['usable']} | {', '.join(q['reasons']) or '—'} |")
    lines += ["", "Scoring2 is documented in `task-sleep_events.json` as manual scoring by scorer 2. The requested `task-sleep_acq-scoring_events.json` describes recording/diary event fields and does not contain a Scoring2 field. Independence/blinding is not established by these sidecars.", ""]
    (RESULTS / "smoke_test_sub-016.md").write_text("\n".join(lines))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Guarded ds005185 PSG preprocessing and YASA baselines")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="run only sub-016 ses-001")
    group.add_argument("--cohort", action="store_true", help="run the size-limited distinct-subject cohort")
    parser.add_argument("--resume", action="store_true", help="skip recordings with both completed cache and JSON log")
    parser.add_argument("--force", action="store_true", help="regenerate selected recording cache/log")
    args = parser.parse_args(argv)
    recordings, sizes = candidates_and_sizes()
    print("All eligible ses-001..004 PSG candidates (exact object sizes, before transfers):", flush=True)
    for candidate in recordings:
        print(f"{candidate['subject']} ses-{candidate['session']}: "
              f"{candidate['set'][0]}={candidate['set'][1]} B; "
              f"{candidate['fdt'][0]}={candidate['fdt'][1]} B", flush=True)
    if args.smoke:
        rec = next((r for r in recordings if r["subject"] == "sub-016" and r["session"] == "001"), None)
        if rec is None: raise RuntimeError("sub-016 ses-001 not found in inventory")
        selected = [rec]
    else:
        selected = select_cohort(recordings)
    print("Pre-download size inventory:", flush=True)
    for r in selected:
        print(f"{r['subject']} ses-{r['session']}: .set {r['set'][1]} B, .fdt {r['fdt'][1]} B; pair {r['raw_bytes']} B", flush=True)
    manifest = download_manifest_text(selected, sizes)
    write_csv(ROOT / "data_manifest.csv", manifest)
    results, meter = run_recordings(selected, sizes, smoke=args.smoke, resume=args.resume,
                                    force=args.force, all_recordings=recordings)
    if args.smoke:
        result = next(x for x in results if x["subject"] == "sub-016")
        write_smoke_summary(result, meter)
        print(json.dumps(json_safe({"epochs": result["n_epochs"], "class_counts": result["class_counts"],
                                    "yasa_scalp": result["yasa_scalp"], "yasa_ear": result["yasa_ear"],
                                    "scoring1_vs_scoring2": result["scoring1_vs_scoring2"],
                                    "channel_quality": result["channel_quality"],
                                    "downloaded_bytes": meter["downloaded_bytes"]}), indent=2), flush=True)
    else:
        write_json(RESULTS / "cohort_preprocess_summary.json", {
            "subjects_used": sorted({r["subject"] for r in results}),
            "recordings_used": len(results), "downloaded_bytes": meter["downloaded_bytes"],
            "peak_disk_bytes": meter["peak_disk_bytes"],
            "total_raw_pair_bytes": sum(r["raw_pair_bytes"] for r in results),
            "excluded_label_counts": {name: sum(x["excluded_label_counts"].get(name, 0) for x in results)
                                      for name in ("A", "Artefact", "Unscored")},
            "recordings": results})


if __name__ == "__main__":
    main()
