# Ear EEG LightGBM evaluation

## Summary

Leave-one-subject-out evaluation covered 29,021 retained epochs from 10 training subjects across 10 folds.
Raw LOSO accuracy was 78.6%, Cohen's kappa 0.697, and macro-F1 0.730.
The 3-epoch median result was accuracy 79.3%, macro-F1 0.733; the 5-epoch result was accuracy 79.6%, macro-F1 0.735.

## Generalization across subjects

Per-subject metrics and mean/std rows are in `per_subject_metrics.csv`.

## Weakest stages and class metrics (raw LOSO)

| Stage | Precision | Recall | F1 | Epochs |
|---|---:|---:|---:|---:|
| Wake | 0.824 | 0.863 | 0.843 | 2405 |
| N1 | 0.339 | 0.440 | 0.383 | 1713 |
| N2 | 0.862 | 0.780 | 0.819 | 13725 |
| N3 | 0.819 | 0.869 | 0.843 | 5914 |
| REM | 0.740 | 0.785 | 0.762 | 5264 |

The weakest raw-prediction class by F1 was **N1**.

## Most important features

| Rank | Feature | Mean gain |
|---:|---|---:|
| 1 | beta_relative_power_roll5_mean | 66518.579 |
| 2 | line_length_uv_per_s | 65939.610 |
| 3 | beta_power_uv2 | 61669.470 |
| 4 | sigma_power_uv2 | 27994.247 |
| 5 | alpha_power_uv2 | 21015.154 |
| 6 | beta_relative_power | 18962.835 |
| 7 | theta_relative_power_roll5_mean | 16831.321 |
| 8 | sigma_relative_power_roll5_mean | 16747.120 |
| 9 | theta_power_uv2 | 16654.723 |
| 10 | beta_relative_power_lag1 | 13327.284 |
| 11 | beta_relative_power_roll5_std | 13320.782 |
| 12 | sigma_relative_power_lead4 | 9917.309 |
| 13 | theta_relative_power_lag1 | 9108.219 |
| 14 | sigma_relative_power_roll5_std | 8922.744 |
| 15 | theta_alpha_ratio_roll5_mean | 8538.477 |

**Warning:** a time-based or raw-amplitude feature appears in the top three: beta_power_uv2.

## Held-out comparison: sub-016 session 001

All rows below use the same 690 Scoring1-labeled epochs as the stored YASA baseline. YASA results were read from the existing JSON and not recomputed.

| Model | Signal | Epochs | Accuracy | Error |
|---|---|---:|---:|---:|
| YASA pretrained | In-ear ELE−ERE | 690 | 18.0% | 82.0% |
| LightGBM | In-ear ELE−ERE | 690 | 68.3% | 31.7% |
| YASA pretrained | Scalp C3−M2 | 690 | 88.0% | 12.0% |

No cohort-wide YASA scores are available; only sub-016 session 001 has stored YASA results.

## Caveats

- Four of the 40 selected nights were excluded: three had no finite ear-channel data and one had a mismatched EEGLAB header/data file. Several retained nights have high signal-invalid rates; see `run_config.json` for per-night counts.
- The cohort CSV was already filtered by the dataset builder. Artifact, unscored, and signal-invalid rows were unavailable when calculating per-night robust scaling and context features; available rows were scaled before the model-stage label filter.
- Centered windows and lead features use future epochs. These results describe offline staging, not real-time prediction.
- The sub-016 evaluation is a single held-out night and should be treated as a demonstration, not an independent validation study.
