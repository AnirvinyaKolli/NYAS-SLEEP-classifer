# Handoff: ear-EEG sleep-stage proof of concept

## Status

The repository contains a consolidated, size-guarded preprocessing and YASA baseline pipeline. The files compile. A fresh smoke test was not run after consolidation by request; `results/prior_cache_sub-016.md` preserves metrics inspected from an earlier cache and clearly labels them as prior diagnostic data. Do not present those figures as verification of the consolidated code.

The goal is a three-way comparison: YASA on scalp PSG, YASA on ear EEG, and a future subject-wise ear-EEG model, using OpenNeuro ds005185. Only sub-016 ses-001 is confirmed readable. No cohort run has started.

## Earlier sub-016 diagnostic

The prior cache had 701 epochs. YASA-scalp kappa was 0.860 (above the 0.5 catastrophic-error threshold); YASA-ear kappa was 0.119. Scoring1/Scoring2 kappa was 0.917 over 690 jointly valid epochs. The runtime mapping matched the expected values, both earlier YASA outputs had 701 entries, and ear predictions were non-empty. N1 occurred 20 times (2.9%); its F1 is unreliable. Full metrics and confusion matrices are in `results/prior_cache_sub-016.md`.

Scoring2 is described in `task-sleep_events.json` as manual sleep scoring by scorer 2. `task-sleep_acq-scoring_events.json` describes recording/diary event fields and does not contain Scoring2. Independence/blinding is not established by these sidecars.

## Fixes consolidated in `src/sleep_pipeline.py`

- Parse and print label codes at runtime from `task-sleep_events.json`, the sidecar containing Scoring1/Scoring2 levels.
- Build the ear YASA signal from every label-aligned epoch.
- Recursively convert NumPy values and keys before JSON serialization; atomically write each per-recording log after cache creation.

The pipeline builds epochs from each Scoring1 onset, crops the scalp YASA signal to the first onset and exact scored duration, asserts prediction/label length, checks all 12 ear channels before filtering, and uses fixed a priori derivations. Cohort resume skips only when both cache and per-recording log exist. Raw pairs are deleted only after a completed cache/log.

## Data and storage

Dataset files, raw downloads, caches, and temporary MNE/YASA installs were removed from the workspace intentionally. Re-pull with `DATA.md`; do not commit data files. The 18 GB transfer budget and 20 GB hard disk guard are in `config.json` and enforced by the pipeline.

Removed during cleanup: the interrupted ses-002 `.set` and partial `.fdt`, the old 8.8 MB epoch cache, stale `step1_2_skipped.json`, superseded `scripts/step1_2_preprocess_yasa.py`, generated Python bytecode, and temporary package installations under `/tmp`.

## Not done

- Fresh smoke test of the consolidated source and ear-channel QC results.
- Cohort run (target 12–14 subjects, one night each, ses-001 preferred, 18 GB cap).
- LightGBM training, subject-wise evaluation, collapsed-stage/binary results, figures, and feature importance.
- `sleep_stager.py`, model binaries, latency measurement, and slide-ready summary.
- Synthetic helper tests are present but were not run. No clinical stimulation policy is defined.

## Slide limitations

- Sleep labels come from scalp PSG; ear predictions are compared against those labels.
- N1 is rare (~3% in the inspected night), so its F1 is unreliable.
- The cohort and per-fold estimates will be small/noisy; use subject-wise splits only.
- Ear derivation selection must be fixed a priori or selected within training folds.
- Scoring1/Scoring2 agreement is not a proven expert ceiling until scorer independence is confirmed.
- YASA's model emitted a scikit-learn version warning in the temporary environment; predictions completed, but compatibility should be reviewed.
- This proof of concept is not a clinically validated sleep-stage classifier or stimulation controller.
