#!/usr/bin/env python3
"""Fetch selected ds005185 PSG recordings and their Scoring1 labels."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())
BUCKET = "s3://openneuro.org/ds005185/"


def inventory() -> dict[str, int]:
    result = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", "--recursive", BUCKET],
        capture_output=True, text=True, check=True,
    )
    sizes = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\s*\d{4}-\d\d-\d\d\s+\d\d:\d\d:\d\d\s+(\d+)\s+(.+)$", line)
        if match:
            sizes[match.group(2)] = int(match.group(1))
    return sizes


def download(key: str, target: Path, expected: int, downloaded: int) -> int:
    if target.exists() and target.stat().st_size == expected:
        print(f"Already present: {target.relative_to(ROOT)}")
        return downloaded
    if downloaded + expected > int(CONFIG["max_download_bytes"]):
        raise RuntimeError(f"Download cap reached before {key}")
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", f"s3://openneuro.org/{key}", str(target)],
        capture_output=True, text=True,
    )
    actual = target.stat().st_size if target.exists() else 0
    if result.returncode or actual != expected:
        target.unlink(missing_ok=True)
        raise IOError(f"Incomplete download {key}: expected {expected}, got {actual}. {result.stderr[-500:]}")
    print(f"Downloaded {target.relative_to(ROOT)} ({actual:,} bytes)")
    return downloaded + actual


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch inputs for in-ear EEG sleep modeling")
    parser.add_argument("--recording", action="append", default=[], metavar="SUBJECT:SESSION",
                        help="recording selector; repeat to fetch a batch")
    args = parser.parse_args()
    recordings = args.recording or ["sub-016:001"]
    files = {}
    for recording in recordings:
        try:
            subject, session = recording.split(":", 1)
            if not re.fullmatch(r"sub-\d+", subject) or not re.fullmatch(r"\d{3}", session):
                raise ValueError
        except ValueError as exc:
            raise SystemExit("Use --recording SUBJECT:SESSION, e.g. sub-016:001") from exc
        stem = f"{subject}_ses-{session}_task-sleep_acq-PSG_eeg"
        prefix = f"ds005185/{subject}/ses-{session}/eeg/"
        recording_files = (
            (prefix + stem + ".set", ROOT / "data/raw" / (stem + ".set")),
            (prefix + stem + ".fdt", ROOT / "data/raw" / (stem + ".fdt")),
            (prefix + f"{subject}_ses-{session}_task-sleep_acq-scoring1_events.tsv",
             ROOT / "data/metadata" / f"{subject}_ses-{session}_task-sleep_acq-scoring1_events.tsv"),
        )
        files.update({key: target for key, target in recording_files})
    files["ds005185/task-sleep_events.json"] = ROOT / "data/metadata/task-sleep_events.json"

    sizes = inventory()
    missing = [key for key in files if key not in sizes]
    if missing:
        raise FileNotFoundError("Recording or label metadata missing from inventory: " + ", ".join(missing))
    needed_bytes = sum(sizes[key] for key, target in files.items()
                       if not target.exists() or target.stat().st_size != sizes[key])
    if needed_bytes > int(CONFIG["max_download_bytes"]):
        raise RuntimeError(
            f"This batch needs {needed_bytes:,} bytes, over the "
            f"{int(CONFIG['max_download_bytes']):,}-byte cap. Split it into smaller batches."
        )
    downloaded = 0
    for key, target in files.items():
        downloaded = download(key, target, sizes[key], downloaded)
    print(f"Fetch complete for {len(recordings)} recording(s). Downloaded this run: {downloaded:,} bytes.")


if __name__ == "__main__":
    main()
