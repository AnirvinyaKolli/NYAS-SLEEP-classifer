# Repository Guidelines

## Project Structure

- `src/ear_training_data.py` extracts ELE−ERE ear EEG features and paired Scoring1 labels for one PSG night.
- `src/yasa_pipeline.py` runs YASA comparisons on local ear and scalp PSG signals.
- `scripts/fetch_data.py` fetches selected OpenNeuro recordings and metadata; `scripts/build_ear_training_dataset.py` builds and combines the fixed cohort in per-night batches.
- `config.json` holds sampling, epoch, amplitude, and download limits. `reports/` documents data and processing; `results/yasa/` stores comparison outputs.
- `data/` is runtime input/output and should not be committed.

## Development and Run Commands

Use Python 3 and a project virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

The fetch script requires the AWS CLI for unsigned access to the public OpenNeuro S3 bucket. Fetch one recording with `python3 scripts/fetch_data.py --recording sub-016:001`. Run the YASA comparison with `python3 src/yasa_pipeline.py --recording sub-016:001`. Build the selected 40-night cohort with `python3 scripts/build_ear_training_dataset.py`; this downloads and processes data, so review `config.json` and available disk/network capacity first. There is no separate build system or dependency lockfile in the repository.

## Code Style

Follow the existing Python style: four-space indentation, `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for constants, type hints on public functions, and short docstrings describing inputs and outputs. Use `pathlib.Path` for paths and keep dataset-specific settings in `config.json`. Preserve subject, session, and epoch identifiers when changing feature or label processing.

## Testing

No automated test suite or test configuration is currently present. For processing changes, check a single known recording before starting a cohort build; compare feature and label epoch keys, verify finite feature values, and inspect stage-wise summaries. Do not use the cohort builder as a quick test because it performs large downloads.

## Commits and Pull Requests

The available Git history contains only an initial commit, so no commit convention is established. Use a short imperative subject (for example, `Center epoch amplitude features`). In pull requests, describe the data or code affected, list commands run, note any download or output changes, and include a small summary or sample when results change.

## Data and Configuration

Never commit downloaded EEG recordings or generated cohort tables. Keep raw recordings under `data/raw/`, metadata under `data/metadata/`, and generated cohort outputs under `data/processed/ear_sleep_training/`. Record changes to cohort selection or filtering in the relevant report and manifest behavior.
