#!/usr/bin/env python3
"""Inventory and explicitly fetch a chosen subset of ds005185 (never the full set)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sleep_pipeline import (CACHE, CONFIG, DATA, META, RAW, RESULTS, candidates_and_sizes,
                                download, download_manifest_text, recording_keys, select_cohort,
                                write_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", action="append", default=[], metavar="SUBJECT:SESSION",
                        help="explicit recording selector, e.g. sub-016:001 (repeatable)")
    parser.add_argument("--download-raw", action="store_true", help="download only explicitly selected .set/.fdt files")
    parser.add_argument("--download-sidecars", action="store_true", help="download only the listed small sidecars")
    parser.add_argument("--manifest", action="store_true", help="write data_manifest.csv from the public inventory")
    args = parser.parse_args()
    recordings, sizes = candidates_and_sizes()
    for r in recordings:
        if r["session"] == "001":
            print(f"INVENTORY {r['subject']} ses-{r['session']} .set={r['set'][1]} .fdt={r['fdt'][1]} pair={r['raw_bytes']} bytes")
    if args.manifest:
        write_csv(ROOT / "data_manifest.csv", download_manifest_text([r for r in recordings if r["session"] == "001"], sizes))
        print(f"Wrote manifest with all ses-001 candidates; sub-016 is the only confirmed-readable recording.")
    selected_ids = set(args.recording)
    selected = [r for r in recordings if f"{r['subject']}:{r['session']}" in selected_ids]
    available = {f"{r['subject']}:{r['session']}" for r in selected}
    missing = selected_ids - available
    if missing:
        raise SystemExit(f"not found in ses-001..004 inventory: {sorted(missing)}")
    if args.download_raw and not selected:
        raise SystemExit("Refusing raw downloads without an explicit --recording subject:session selector")
    meter = {"downloaded_bytes": 0, "peak_disk_bytes": 0}
    for r in selected:
        if args.download_sidecars:
            sidecars = {
                "ds005185/dataset_description.json": META / "dataset_description.json",
                "ds005185/participants.tsv": META / "participants.tsv",
                "ds005185/task-sleep_events.json": META / "task-sleep_events.json",
                "ds005185/task-sleep_acq-scoring_events.json": META / "task-sleep_acq-scoring_events.json",
                "ds005185/acq-PSG_channels.tsv": META / "acq-PSG_channels.tsv",
                **{key: META / Path(key).name for key in recording_keys(r).values()},
            }
            for key, target in sidecars.items():
                print(f"SIDECAR INVENTORY {key} {sizes[key]} bytes")
            for key, target in sidecars.items():
                download(key, target, sizes[key], meter)
        if args.download_raw:
            for ext in ("set", "fdt"):
                key, n = r[ext]
                download(key, RAW / Path(key).name, n, meter)
    if not args.download_raw and not args.download_sidecars:
        print("Inventory only. To fetch the smoke test, pass --recording sub-016:001 --download-sidecars --download-raw.")
    print(f"Downloaded this invocation: {meter['downloaded_bytes']} bytes; peak data/results usage: {meter['peak_disk_bytes']} bytes")


if __name__ == "__main__":
    main()
