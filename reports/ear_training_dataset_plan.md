# Ear-EEG training dataset: cohort and extraction plan

## Cohort selection

The fixed cohort is a reproducible sample of 10 ds005185 participants: **sub-001, sub-002, sub-003, sub-004, sub-008, sub-009, sub-011, sub-012, sub-017, and sub-018**. The list was selected with random seed 42 from the 20 numbered participants; selection does not depend on model scores or label distributions. The batch job uses sessions **001, 002, 003, and 004 only**. The dataset README identifies these as the PSG nights with ear EEG and scalp PSG recorded together and scored. Later ear-only sessions are excluded because they lack the paired scalp scoring key. [Dataset README](https://github.com/OpenNeuroDatasets/ds005185/blob/main/README)

This is 40 subject-night recordings. The selected list is fixed in the batch script so the run can be reproduced. Change its `SUBJECTS` constant to select another cohort.

The batch builder skips and records any night where either ear channel has no finite samples, since ELE−ERE cannot be formed. This has been found in `sub-003:004`, `sub-004:004`, and `sub-008:003`. `sub-011:004` is also excluded because its EEGLAB header expects 1,740,387,000 bytes of float32 data, while its paired `.fdt` is only 309,815,000 bytes. If no other nights are unusable, the combined table will contain up to 36 nights.

## Reusable one-night function

[`src/ear_training_data.py`](../src/ear_training_data.py) exposes:

```python
from src.ear_training_data import process_one_night

features, labels = process_one_night("sub-016", 1)
```

It returns two pandas tables with matching `subject`, `session`, and `epoch_index` keys. Each row represents one 30-second Scoring1 epoch, with `onset_seconds` and `duration_seconds` included in both tables. Features describe ELE−ERE. Labels contain the integer `stage_code`, readable `stage_name`, and `is_sleep_stage` flag. Epoch indices start at 1 within each night. The function only accepts sessions 001–004 and checks the input files, label columns, timing, channel presence, and signal bounds.

Scored epochs that are incomplete because they begin before or end after the EEG recording are skipped with a warning; an out-of-bounds epoch between valid epochs remains a structural error.

## Per-epoch ear measurements

MNE reads only the ELE and ERE contacts from each PSG EEGLAB recording. The signal is resampled to 100 Hz, converted from MNE volts to microvolts, and derived as:

```text
ear_signal(t) = ELE(t) − ERE(t)
```

For every 30-second epoch, the function computes 27 measurements:

| Feature family | Measurements |
|---|---|
| Amplitude and distribution | Mean, median, standard deviation, RMS, peak-to-peak range, 95th percentile absolute amplitude, skewness, excess kurtosis |
| Signal shape | Zero-crossing rate, line length, Hjorth activity, Hjorth mobility, Hjorth complexity |
| Frequency power | Absolute and relative delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), sigma (12–16 Hz), beta (16–30 Hz) power |
| Frequency summary | Delta/beta and theta/alpha power ratios, normalized spectral entropy, peak frequency from 0.5–30 Hz |

Absolute band power is in µV². Per-epoch power is estimated with Welch's method using a 5-second window. These are hand-engineered candidate inputs for a future LightGBM model; they are not YASA's internal features or predictions.

The source EEGLAB data can contain NaN samples marking rejected channel data. The function records the original missing-sample mask, linearly interpolates only so MNE can resample, and computes finite features for every epoch. It flags `signal_valid == False` when an epoch overlaps any original missing samples or its peak-to-peak amplitude exceeds the configured 1,000 µV limit. `nonfinite_sample_fraction`, `peak_abs_uv`, and `amplitude_outlier` make those quality conditions visible. Absolute peak is retained for inspection but is not the outlier criterion because per-epoch DC offsets can be large. Invalid-signal epochs remain in the one-night output for audit but are excluded by the cohort builder after feature calculation.

## Labels and filtering

The function reads the recording-specific `*_acq-scoring1_events.tsv` and maps its codes using `task-sleep_events.json`:

| Code | Stage name | Kept for five-class training? |
|---:|---|---|
| 1 | Wake | Yes |
| 2 | REM | Yes |
| 3 | N1 | Yes |
| 4 | N2 | Yes |
| 5 | N3 | Yes |
| 6 | A | No |
| 7 | Artefact | No |
| 8 | Unscored | No |

`process_one_night` returns measurements and labels for all epochs, including A, Artefact, and Unscored. The batch job computes the complete night first, then excludes signal-invalid epochs and codes 6–8. It logs each night’s source count and exclusions. This preserves the full signal context during feature computation while ensuring those epochs do not enter the final training table.

## Download, process, save, delete, combine

[`scripts/build_ear_training_dataset.py`](../scripts/build_ear_training_dataset.py) reads the OpenNeuro object inventory once. It then handles one night at a time:

1. Download that night's `.set`, `.fdt`, and Scoring1 TSV, verifying their expected sizes.
2. Compute all epoch features and read all Scoring1 labels.
3. Drop signal-invalid, A, Artefact, and Unscored epochs after computation.
4. Save a compact per-night batch and update the manifest.
5. Delete that night's large `.set` and `.fdt` files.
6. After all 40 nights are complete, stack the saved batches into one table.

Output files:

```text
data/processed/ear_sleep_training/
├── ear_sleep_training.csv
├── manifest.json
└── by_night/
    ├── sub-001_ses-001.csv
    └── ...
```

The combined CSV has one row per usable labeled epoch, including subject, session, epoch key, the 27 ear measurements, and Scoring1 target columns. The manifest records cohort selection, transfer totals, per-night kept/dropped counts, and the final row count. The per-night CSVs are small checkpoints. The configured 18 GB transfer cap applies to each individual night; total transfer across 40 nights can exceed that because raw EEG is removed after each saved batch.

Run the batch job with:

```bash
python scripts/build_ear_training_dataset.py
```

The job stops on the first structural or processing failure. It does not retry or silently skip the failed night. It records the error in the manifest and leaves any raw files already downloaded for diagnosis. A subsequent run is blocked until the cause is fixed and the failed-run manifest is deliberately cleared.

The script does not train LightGBM. When training is added, split by **subject**, not random epochs, so a participant's other nights cannot leak into validation data.

## Local one-night check

Only sub-016 ses-001 is currently present in the workspace; it is outside the default cohort. It can be extracted directly:

```python
features, labels = process_one_night("sub-016", "001")
```

That run returned 701 rows in each table, matching epoch keys, and finite values for all 27 measurements. Seven epochs overlapped source NaNs; interpolation was used only for resampling and those rows were flagged. All 11 code-7 Artefact labels remained in the one-night output and were marked `is_sleep_stage == False`.

The initial absolute-peak check exposed a large per-epoch baseline offset: median `mean_uv` was about −15,733 µV while median peak-to-peak amplitude was about 176 µV. This is why absolute peak alone incorrectly marked most epochs unusable. MNE provides EEG data in SI volts and the extractor converts the bipolar signal to microvolts once. The current outlier rule uses peak-to-peak amplitude (1,000 µV limit), which is invariant to a constant offset. Stage-direction checks from the initial extraction (before that outlier rule) showed median relative delta power of 0.695 for Wake and 0.930 for N3; median beta power was 5.14 µV² for Wake and 1.27 µV² for N3; median Hjorth activity was 7,948 µV² for Wake and 835 µV² for N3.
