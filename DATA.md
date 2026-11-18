# Data access and storage

Dataset: OpenNeuro [ds005185 v1.0.2](https://openneuro.org/datasets/ds005185/versions/1.0.2), EESM19, CC0. Sessions ses-001 through ses-004 contain simultaneous ear EEG and PSG scoring; later sessions are ear-only and are excluded.

## Inventory and fetch

This machine used AWS CLI v2 against OpenNeuro's public S3 bucket:

```bash
aws s3 ls --no-sign-request --recursive s3://openneuro.org/ds005185/
python scripts/fetch_data.py --manifest
```

The first command is inventory only. `fetch_data.py` also inventories and prints exact byte sizes before any transfer. Fetch only the smoke-test recording and its sidecars with:

```bash
python scripts/fetch_data.py --recording sub-016:001 --download-sidecars --download-raw
```

The common sidecars are `dataset_description.json`, `participants.tsv`, `task-sleep_events.json`, `task-sleep_acq-scoring_events.json`, and `acq-PSG_channels.tsv`; each recording also needs `*_task-sleep_acq-scoring1_events.tsv` and `*_task-sleep_acq-scoring2_events.tsv`. The PSG signal is the matching `*_task-sleep_acq-PSG_eeg.set` and `.fdt` pair. `data_manifest.csv` records expected sizes and status: sub-016 ses-001 is the only night confirmed readable so far; other candidates are not yet selected or verified.

AWS CLI fallback for one explicit recording, after checking its size in the inventory:

```bash
aws s3 cp --no-sign-request s3://openneuro.org/ds005185/sub-016/ses-001/eeg/sub-016_ses-001_task-sleep_acq-PSG_eeg.set data/raw/
aws s3 cp --no-sign-request s3://openneuro.org/ds005185/sub-016/ses-001/eeg/sub-016_ses-001_task-sleep_acq-PSG_eeg.fdt data/raw/
```

Alternatively, with DataLad installed:

```bash
datalad install -s https://github.com/OpenNeuroDatasets/ds005185.git data/ds005185
cd data/ds005185
datalad get dataset_description.json participants.tsv task-sleep_events.json task-sleep_acq-scoring_events.json acq-PSG_channels.tsv sub-016/ses-001/eeg/sub-016_ses-001_task-sleep_acq-scoring1_events.tsv sub-016/ses-001/eeg/sub-016_ses-001_task-sleep_acq-scoring2_events.tsv sub-016/ses-001/eeg/sub-016_ses-001_task-sleep_acq-PSG_eeg.set sub-016/ses-001/eeg/sub-016_ses-001_task-sleep_acq-PSG_eeg.fdt
```

The tested transport was AWS S3. Network access is required for the fetch step; the sandbox used for this work restricts network by default. Files may also be pre-downloaded into `data/raw` and `data/metadata`.

## Budget and rebuild

The inspected sub-016 ses-001 raw pair is **1,160,095,568 bytes**. Cohort downloads stop at **18,000,000,000 bytes**, preserving a 2 GB margin below the 20 GB hard disk limit. Processing is one recording at a time: download, load and preprocess, write compact `.npz` epochs and predictions, write a per-recording JSON log, then delete the raw pair. Transfers are checked against the inventoried object size; source-truncated or unreadable recordings are logged and skipped, with a later PSG session attempted when available and within budget.

Regenerate caches with the single resume command in README:

```bash
python scripts/run_pipeline.py --cohort --resume
```

One sub-016 transfer took roughly a minute at the observed connection rate; preprocessing and the two CPU YASA passes added about another minute. This is a rough observation, not a cohort runtime estimate. The other candidate recordings have not been confirmed readable.
