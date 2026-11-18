# Ear-EEG sleep-staging proof of concept

This repository contains a guarded ds005185 preprocessing and YASA baseline pipeline. The cohort run is intentionally not included in this handoff's results.

## Setup

Use Python 3.14 and install the pinned packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install AWS CLI v2 separately and confirm `aws --version` works. Public reads use `--no-sign-request`; no AWS credentials are needed. The process needs network access to OpenNeuro. See [DATA.md](DATA.md).

## Resume the cohort preprocessing

```bash
python scripts/run_pipeline.py --cohort --resume
```

It inventories eligible PSG file sizes before downloading, selects one ses-001 night per subject (sub-016 first), enforces the 18 GB transfer and 20 GB disk caps, skips a recording only when both its cache and per-recording JSON log exist, and writes progress after each finished recording. It prints subject, session, epoch count, scalp/ear kappa, and the running byte tally. Raw pairs are deleted after a recording is cached and logged. Truncated or unreadable ses-001 files are logged; later PSG sessions are attempted within the remaining cap.

To repeat the one-recording sanity run:

```bash
python scripts/run_pipeline.py --smoke --force
```

The prior-cache diagnostic in `results/prior_cache_sub-016.md` is not a fresh smoke test.

## Current status

The consolidated source files compile. The synthetic helper tests are present but were not run. A fresh smoke test, cohort preprocessing, model training, subject-wise evaluation, figures, and deployment interface remain for the teammate.
