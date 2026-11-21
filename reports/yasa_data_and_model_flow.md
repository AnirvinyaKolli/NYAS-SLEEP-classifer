# Dataset structure and current YASA data flow

## Purpose and scope

This report documents how this project currently reads the OpenNeuro **ds005185 (EESM19: Ear-EEG Sleep Monitoring 2019)** dataset, selects one recording, and passes its signals to YASA. It describes the current comparison pipeline; it does not describe the planned in-ear LightGBM training pipeline.

The local example is **sub-016, session 001**. Session 001 is one of the dataset's PSG nights, where ear EEG and partial PSG were recorded together. The dataset includes 20 participants. The public dataset description says participants each contributed four PSG nights; 10 participants also contributed up to 12 additional ear-EEG-only nights. Each night is a separate BIDS session. The sessions after the PSG nights do not all have the scalp PSG channels used in this project's comparison. [Dataset README](https://github.com/OpenNeuroDatasets/ds005185/blob/main/README)

## How the database is organized

The dataset uses BIDS-style subject/session folders. At the top level it has shared dataset and participant metadata, task-level JSON sidecars, and channel descriptions. Under a subject and session, EEG and behavioral files are grouped by modality. In simplified form:

```text
ds005185/
├── dataset_description.json
├── participants.tsv / participants.json
├── task-sleep_events.json
├── task-sleep_acq-scoring_events.json
├── acq-PSG_channels.tsv
├── acq-earEEG_channels.tsv
├── sub-016/
│   └── ses-001/
│       ├── eeg/
│       │   ├── sub-016_ses-001_task-sleep_acq-PSG_eeg.set
│       │   ├── sub-016_ses-001_task-sleep_acq-PSG_eeg.fdt
│       │   ├── sub-016_ses-001_task-sleep_acq-scoring1_events.tsv
│       │   └── sub-016_ses-001_task-sleep_acq-scoring2_events.tsv
│       └── beh/
│           ├── diary / actigraphy files
│           └── comfort questionnaire files
└── other subjects and sessions...
```

This is an illustrative subset, not a complete listing of every file. The full dataset also contains calibration and auditory steady-state response recordings. The dataset README describes sleep recordings, diaries, actigraphy, questionnaires, and electrode checks. [Dataset README](https://github.com/OpenNeuroDatasets/ds005185/blob/main/README)

### EEG signal files

The PSG sleep recording is stored as an EEGLAB `.set` file paired with an `.fdt` data file. The `.set` file contains the EEGLAB dataset structure and recording/channel metadata; the `.fdt` file is its companion continuous signal data. The local pair for sub-016 ses-001 is about 101 MB (`.set`) and 1006 MB (`.fdt`). The public channel sidecar describes 25 channels sampled at 500 Hz, with values in microvolts and an average reference: 12 ear EEG contacts (ELA, ELB, ELC, ELT, ELE, ELI, ERA, ERB, ERC, ERT, ERE, ERI), eight scalp EEG channels (M1, F3, C3, O1, M2, F4, C4, O2), two EOG channels, and three EMG channels. [PSG channel sidecar](https://github.com/OpenNeuroDatasets/ds005185/blob/main/acq-PSG_channels.tsv)

The `acq-PSG` recording therefore contains both the ear contacts and scalp PSG channels. The current comparison does not need a separate ear-only recording to compare those signal locations.

### Sleep-stage labels

The recording-specific event TSV contains one row per manually scored epoch. For the local recording it has 701 data rows and three columns:

| Column | Meaning |
|---|---|
| `onset` | Epoch start time in seconds from the recording start |
| `duration` | Epoch length; 30 seconds in this file |
| `Scoring1` | Sleep-stage code assigned by scorer 1 |

The shared `task-sleep_events.json` maps the codes: 1 Wake, 2 REM, 3 N1, 4 N2, 5 N3, 6 A, 7 Artefact, and 8 Unscored. The same sidecar documents Scoring2, but the current fetch and YASA pipeline use Scoring1 only. Scoring2 is not used for training or for the reported comparison.

For sub-016 ses-001, the 701 labels are Wake 199, REM 115, N1 20, N2 205, N3 151, Artefact 11, and zero A or Unscored. The current metrics exclude codes 6–8 and include the 690 epochs labeled 1–5.

## What the fetch script downloads

[`scripts/fetch_data.py`](../scripts/fetch_data.py) first lists the object inventory for ds005185 from the public OpenNeuro S3 bucket. That operation reads file names and sizes across the dataset; it does not download all EEG recordings. It then fetches four objects for the selected subject/session:

1. The PSG `.set` file.
2. Its `.fdt` companion.
3. That recording's Scoring1 event TSV.
4. The shared `task-sleep_events.json` label mapping.

Transfers are size-checked, and a configured 18 GB per-invocation download cap is enforced. Existing local files are reused when their sizes match the inventory. The fetch script does not fetch Scoring2, participant demographics, diaries, actigraphy, or other recordings.

## How the current YASA pipeline prepares model input

[`src/yasa_pipeline.py`](../src/yasa_pipeline.py) requires those four local files, reads Scoring1 labels/onsets/durations, and loads the `.set`/`.fdt` recording through MNE's EEGLAB reader. It checks for the four channels needed by the current comparison: ELE, ERE, C3, and M2.

The processing steps are:

1. **Select channels.** The pipeline drops every channel except ELE, ERE, C3, and M2. EOG, EMG, other ear contacts, and other scalp leads are present in the source PSG recording but are not sent to YASA.
2. **Load and resample.** MNE loads the selected channels and resamples them from the recording's 500 Hz to 100 Hz. The current code applies no additional band-pass or notch filter.
3. **Align the scored interval.** The pipeline rounds each TSV onset to a sample at 100 Hz, checks the epoch durations and recording bounds, and extracts the continuous interval from the first scored onset through the last scored epoch. For this file the first onset is approximately 26 seconds. It checks that the extracted interval spans exactly 701 consecutive 30-second epochs.
4. **Create two bipolar signals.** At each sample it subtracts the negative channel from the positive channel:
   - In-ear: `ELE − ERE`
   - Scalp PSG: `C3 − M2`
5. **Call YASA separately.** Each signal becomes a one-channel MNE `RawArray` at 100 Hz with channel type EEG. YASA receives one EEG channel and no EOG, EMG, age, or sex metadata. Its pretrained `SleepStaging` model predicts one label per 30-second epoch. The model probabilities are not currently saved or analyzed.
6. **Map and score predictions.** YASA's labels (`WAKE`, `REM`, `N1`, `N2`, `N3`) are mapped back to the event-file codes. Predictions are compared epoch-by-epoch with Scoring1. The report calculates overall error and accuracy, error per manually labeled stage, the stage with the highest error, predicted stage counts, and the in-ear minus scalp error gap.

YASA documents that its staging features assume microvolt EEG data, that MNE signals are automatically converted to microvolts when passed as `Raw`, and that its default classifiers are trained from PSG recordings. It recommends central scalp EEG referenced to mastoids (such as C3−M2/C4−M1) and says not to prefilter the signal before staging. This pipeline uses C3−M2 as a conventional scalp reference and currently leaves the additional filtering to YASA. [YASA `SleepStaging` documentation](https://github.com/raphaelvallat/yasa/blob/master/src/yasa/staging.py)

## Current local result

The existing report is [`results/yasa/sub-016_ses-001_comparison.md`](../results/yasa/sub-016_ses-001_comparison.md), with machine-readable values in the adjacent JSON file. It compares both inputs on the same 690 valid Scoring1 epochs:

| Input to pretrained YASA | Errors | Error rate | Accuracy |
|---|---:|---:|---:|
| In-ear ELE−ERE | 566 / 690 | 82.0% | 18.0% |
| Scalp PSG C3−M2 | 83 / 690 | 12.0% | 88.0% |

The scalp PSG model's largest per-stage error is N1: 9 of 20 labeled N1 epochs (45%). The in-ear model predicts mostly N1 and predicts no REM or N3 in this recording. These are agreement statistics against one human scoring column, not an independent measurement of physiological truth. YASA's FAQ reports roughly 85% accuracy against expert-consensus scoring overall and cautions that N1 is its least accurate stage, so a comparatively high N1 error on this small N1 sample is not surprising. [YASA FAQ](https://yasa-sleep.org/faq.html)

## What this pipeline does not do

- It does **not** train or fine-tune a model. Both signals are passed independently to YASA's pretrained classifier.
- It does **not** use Scoring2, EOG, EMG, demographics, actigraphy, diaries, or questionnaire data.
- It does **not** combine the 12 ear contacts into a feature set. The in-ear baseline is only ELE−ERE.
- It does **not** run cross-validation or estimate performance across participants. The saved result is one recording-level comparison.
- Its error rate is disagreement with Scoring1; it is not a clinical validation or a claim that every disagreement is a YASA error.

These boundaries keep the present YASA comparison distinct from the planned in-ear LightGBM work. A LightGBM report should have its own preprocessing, training/validation split, per-subject evaluation, and output directory so its results are not mixed with this pretrained-YASA baseline.
