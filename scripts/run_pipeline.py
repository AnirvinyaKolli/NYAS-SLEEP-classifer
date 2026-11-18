#!/usr/bin/env python3
"""Entry point for the sub-016 smoke test or the later size-capped cohort run."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sleep_pipeline import main

if __name__ == "__main__":
    main()
