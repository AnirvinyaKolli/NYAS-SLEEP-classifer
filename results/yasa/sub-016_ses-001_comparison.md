# YASA signal comparison: sub-016:001

Compared on the same 690 valid Scoring1 epochs.

## Overall

| Signal | Error | Accuracy | Weakest stage | Stage error |
|---|---:|---:|---|---:|
| In-ear EEG (ELE−ERE) | 82.0% | 18.0% | REM | 100.0% |
| Scalp PSG (C3−M2) | 12.0% | 88.0% | N1 | 45.0% |

**In-ear error minus scalp PSG error:** +70.00 percentage points.

## Error by stage

| Stage | Labeled epochs | In-ear error | Scalp PSG error | Ear minus scalp |
|---|---:|---:|---:|---:|
| Wake | 199 | 48.2% | 6.5% | +41.7 pp |
| REM | 115 | 100.0% | 21.7% | +78.3 pp |
| N1 | 20 | 15.0% | 45.0% | -30.0 pp |
| N2 | 205 | 98.0% | 17.6% | +80.5 pp |
| N3 | 151 | 100.0% | 0.0% | +100.0 pp |

YASA predictions for In-ear EEG (ELE−ERE): N1 564, N2 10, WAKE 127.

YASA predictions for Scalp PSG (C3−M2): N1 27, N2 198, N3 171, REM 109, WAKE 196.
