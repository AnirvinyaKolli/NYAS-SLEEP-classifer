"""Build one-night in-ear EEG features paired with manual sleep-stage labels."""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())
DATA = ROOT / "data"
RAW = DATA / "raw"
METADATA = DATA / "metadata"
PSG_SESSIONS = {"001", "002", "003", "004"}
EAR_CHANNELS = ("ELE", "ERE")
SLEEP_STAGES = {1, 2, 3, 4, 5}
BANDS = ("delta", "theta", "alpha", "sigma", "beta")
FEATURE_COLUMNS = (
    "mean_uv", "median_uv", "std_uv", "rms_uv", "peak_to_peak_uv", "abs_q95_uv",
    "skewness", "kurtosis_excess", "zero_crossing_rate_hz", "line_length_uv_per_s",
    "hjorth_activity_uv2", "hjorth_mobility_hz", "hjorth_complexity",
    *(f"{band}_power_uv2" for band in BANDS),
    *(f"{band}_relative_power" for band in BANDS),
    "delta_beta_ratio", "theta_alpha_ratio", "spectral_entropy", "spectral_peak_hz",
)


def _recording_ids(subject: str | int, night: str | int) -> tuple[str, str]:
    subject_text = str(subject)
    if subject_text.isdigit():
        subject_text = f"sub-{int(subject_text):03d}"
    match = re.fullmatch(r"sub-(\d+)", subject_text)
    if not match:
        raise ValueError("subject must be an ID such as 'sub-016' or 16")
    subject_text = f"sub-{int(match.group(1)):03d}"
    try:
        night_text = f"{int(night):03d}"
    except (TypeError, ValueError) as exc:
        raise ValueError("night must be a PSG session number from 1 to 4") from exc
    if night_text not in PSG_SESSIONS:
        raise ValueError("Only sessions 001–004 have the paired PSG sleep scoring used here")
    return subject_text, night_text


def _epoch_features(epoch_uv: np.ndarray, sfreq: float) -> dict[str, float]:
    """Return compact time- and frequency-domain features for one 30 s signal."""
    x = np.asarray(epoch_uv, dtype=float)
    centered = x - np.mean(x)
    variance = float(np.mean(centered ** 2))
    dx = np.diff(x)
    ddx = np.diff(dx)
    dx_var = float(np.mean(dx ** 2)) if dx.size else 0.0
    ddx_var = float(np.mean(ddx ** 2)) if ddx.size else 0.0
    eps = np.finfo(float).eps

    features = {
        "mean_uv": float(np.mean(centered)),
        "median_uv": float(np.median(centered)),
        "std_uv": float(np.std(x)),
        "rms_uv": float(np.sqrt(np.mean(centered ** 2))),
        "peak_to_peak_uv": float(np.ptp(x)),
        "abs_q95_uv": float(np.quantile(np.abs(centered), 0.95)),
        "skewness": float(skew(x, bias=False)) if variance > eps else 0.0,
        "kurtosis_excess": float(kurtosis(x, fisher=True, bias=False)) if variance > eps else 0.0,
        "zero_crossing_rate_hz": float(np.count_nonzero(np.diff(np.signbit(centered))) / (x.size / sfreq)),
        "line_length_uv_per_s": float(np.sum(np.abs(dx)) / (x.size / sfreq)),
        "hjorth_activity_uv2": variance,
        "hjorth_mobility_hz": float(np.sqrt(dx_var / max(variance, eps)) * sfreq),
        "hjorth_complexity": float(
            np.sqrt(ddx_var / max(dx_var, eps)) / max(np.sqrt(dx_var / max(variance, eps)), eps)
        ),
    }

    freq, psd = welch(x, fs=sfreq, nperseg=min(int(5 * sfreq), x.size))
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "sigma": (12.0, 16.0),
        "beta": (16.0, 30.0),
    }
    band_power = {}
    for name, (low, high) in bands.items():
        mask = (freq >= low) & (freq < high)
        power = float(trapezoid(psd[mask], freq[mask])) if mask.sum() > 1 else 0.0
        band_power[name] = power
        features[f"{name}_power_uv2"] = power
    total_power = sum(band_power.values())
    for name, power in band_power.items():
        features[f"{name}_relative_power"] = float(power / max(total_power, eps))
    features["delta_beta_ratio"] = float(band_power["delta"] / max(band_power["beta"], eps))
    features["theta_alpha_ratio"] = float(band_power["theta"] / max(band_power["alpha"], eps))

    spectral_mask = (freq >= 0.5) & (freq < 30.0)
    spectral_power = psd[spectral_mask]
    spectral_freq = freq[spectral_mask]
    if spectral_power.size and float(spectral_power.sum()) > eps:
        probabilities = spectral_power / spectral_power.sum()
        features["spectral_entropy"] = float(
            -np.sum(probabilities * np.log(probabilities + eps)) / np.log(probabilities.size)
        ) if probabilities.size > 1 else 0.0
        features["spectral_peak_hz"] = float(spectral_freq[np.argmax(spectral_power)])
    else:
        features["spectral_entropy"] = 0.0
        features["spectral_peak_hz"] = 0.0
    return features


def process_one_night(subject: str | int, night: str | int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ear-feature table, manual-label table) for one PSG night.

    Args:
        subject: BIDS subject ID (for example ``"sub-016"`` or ``16``).
        night: PSG session number 1–4 (for example ``1`` or ``"001"``).

    Each output row represents one 30-second epoch. Join the returned tables on
    ``subject``, ``session``, and ``epoch_index``. The label table retains artifact
    and unscored labels; filter to stage codes 1–5 when fitting a sleep-stage model.
    The ear signal is the bipolar derivation ELE−ERE, matching the current YASA
    in-ear comparison. ``signal_valid`` is false when an epoch contains source
    NaNs or exceeds the configured peak-to-peak amplitude limit. Such epochs retain
    finite measurements for audit, but should be excluded from model fitting.
    """
    subject_id, session_id = _recording_ids(subject, night)
    stem = f"{subject_id}_ses-{session_id}_task-sleep_acq-PSG_eeg"
    set_path = RAW / f"{stem}.set"
    fdt_path = RAW / f"{stem}.fdt"
    scoring_path = METADATA / f"{subject_id}_ses-{session_id}_task-sleep_acq-scoring1_events.tsv"
    mapping_path = METADATA / "task-sleep_events.json"
    absent = [path for path in (set_path, fdt_path, scoring_path, mapping_path) if not path.exists()]
    if absent:
        message = ", ".join(str(path.relative_to(ROOT)) for path in absent)
        raise FileNotFoundError(f"Missing local PSG input(s): {message}. Fetch with scripts/fetch_data.py.")

    table = pd.read_csv(scoring_path, sep="\t")
    required_columns = {"onset", "duration", "Scoring1"}
    if not required_columns.issubset(table.columns):
        raise ValueError(f"Scoring file must contain columns {sorted(required_columns)}")
    onsets = table["onset"].astype(float).to_numpy()
    durations = table["duration"].astype(float).to_numpy()
    labels = table["Scoring1"].astype(int).to_numpy()
    epoch_seconds = int(CONFIG["epoch_seconds"])
    if not len(labels) or not (len(labels) == len(onsets) == len(durations)):
        raise ValueError("Scoring labels, onsets, and durations must be nonempty and have equal lengths")
    if not np.all(np.isfinite(onsets)) or not np.allclose(durations, epoch_seconds):
        raise ValueError("Scoring file has invalid onset times or non-30-second epoch durations")
    if len(onsets) > 1 and not np.allclose(np.diff(onsets), epoch_seconds, atol=0.02):
        raise ValueError("Scoring epochs are not contiguous 30-second intervals")

    raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose="ERROR")
    missing_channels = sorted(set(EAR_CHANNELS) - set(raw.ch_names))
    if missing_channels:
        raise ValueError("Missing required in-ear channels: " + ", ".join(missing_channels))
    raw.pick(list(EAR_CHANNELS))
    raw.load_data()
    original_sfreq = float(raw.info["sfreq"])
    source = raw.get_data(picks=["ELE", "ERE"])
    finite = np.isfinite(source)
    bad_source_samples = ~finite.all(axis=0)
    if any(not channel.any() for channel in finite):
        raise ValueError("An ear channel contains no finite samples")

    # The distributed EEGLAB files use NaN to mark rejected channel samples.
    # Interpolate only to let MNE resample; keep the original bad-sample mask so
    # affected epochs are flagged and can be omitted from model fitting later.
    time = np.arange(source.shape[1])
    filled = np.vstack([
        np.interp(time, time[channel_finite], channel[channel_finite])
        for channel, channel_finite in zip(source, finite)
    ])
    clean_for_resampling = mne.io.RawArray(filled, raw.info.copy(), verbose="ERROR")
    clean_for_resampling.resample(float(CONFIG["sampling_rate"]), npad="auto", verbose="ERROR")
    sfreq = float(clean_for_resampling.info["sfreq"])
    samples = clean_for_resampling.get_data(picks=["ELE", "ERE"])
    signal_uv = (samples[0] - samples[1]) * 1e6
    samples_per_epoch = int(round(epoch_seconds * sfreq))
    starts = np.rint(onsets * sfreq).astype(int)
    original_samples_per_epoch = int(round(epoch_seconds * original_sfreq))
    original_starts = np.rint(onsets * original_sfreq).astype(int)
    epoch_in_bounds = (
        (starts >= 0)
        & (starts + samples_per_epoch <= signal_uv.size)
        & (original_starts >= 0)
        & (original_starts + original_samples_per_epoch <= source.shape[1])
    )
    out_of_bounds = np.flatnonzero(~epoch_in_bounds)
    if out_of_bounds.size:
        valid_positions = np.flatnonzero(epoch_in_bounds)
        if not valid_positions.size:
            raise ValueError("No complete scored epochs fall within the EEG recording")
        first_valid = int(valid_positions[0])
        last_valid = int(valid_positions[-1])
        if np.any(~epoch_in_bounds[first_valid:last_valid + 1]):
            raise ValueError("A scored epoch falls outside the EEG recording between valid epochs")
        warnings.warn(
            f"Skipping {out_of_bounds.size} incomplete scored epoch(s) at the EEG recording boundary",
            RuntimeWarning,
            stacklevel=2,
        )
    bad_prefix = np.concatenate(([0], np.cumsum(bad_source_samples, dtype=np.int64)))
    bad_counts = np.zeros(len(onsets), dtype=np.int64)
    for epoch_position in np.flatnonzero(epoch_in_bounds):
        start = original_starts[epoch_position]
        bad_counts[epoch_position] = (
            bad_prefix[start + original_samples_per_epoch] - bad_prefix[start]
        )

    feature_rows = []
    label_rows = []
    mapping_doc = json.loads(mapping_path.read_text())
    mapping = {int(code.removeprefix("s")): str(name)
               for code, name in mapping_doc["Scoring1"]["levels"].items()}
    for index, (start, code, onset) in enumerate(zip(starts, labels, onsets), start=1):
        if not epoch_in_bounds[index - 1]:
            continue
        epoch = signal_uv[start:start + samples_per_epoch]
        if epoch.size != samples_per_epoch:
            raise ValueError(f"Epoch {index} has {epoch.size} samples; expected {samples_per_epoch}")
        keys = {
            "subject": subject_id,
            "session": session_id,
            "epoch_index": index,
            "onset_seconds": float(onset),
            "duration_seconds": epoch_seconds,
        }
        nonfinite_fraction = float(bad_counts[index - 1] / original_samples_per_epoch)
        measured = _epoch_features(epoch, sfreq)
        peak_abs_uv = float(np.max(np.abs(epoch - np.mean(epoch))))
        amplitude_outlier = measured["peak_to_peak_uv"] > float(CONFIG["max_peak_to_peak_epoch_uv"])
        signal_valid = bad_counts[index - 1] == 0 and not amplitude_outlier
        feature_rows.append({**keys, "signal_valid": signal_valid,
                             "nonfinite_sample_fraction": nonfinite_fraction,
                             "peak_abs_uv": peak_abs_uv,
                             "amplitude_outlier": amplitude_outlier, **measured})
        label_rows.append({
            **keys,
            "stage_code": int(code),
            "stage_name": mapping.get(int(code), f"Unknown_{int(code)}"),
            "is_sleep_stage": int(code) in SLEEP_STAGES,
        })
    return pd.DataFrame(feature_rows), pd.DataFrame(label_rows)
