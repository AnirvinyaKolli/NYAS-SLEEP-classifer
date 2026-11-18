# Prior sub-016 cache diagnostic (not a fresh smoke test)

Calculated from the pre-existing cache before it was removed. This is evidence from an earlier pipeline attempt, not a fresh run of the consolidated code. The recording had 701 Scoring1 rows; 690 valid stages were evaluated after excluding the 11 Artefact epochs.

- Mapping source: `task-sleep_events.json`
- Mapping: 1=Wake, 2=REM, 3=N1, 4=N2, 5=N3, 6=A, 7=Artefact, 8=Unscored
- Counts: Wake 199, REM 115, N1 20, N2 205, N3 151, A 0, Artefact 11, Unscored 0
- YASA-scalp: accuracy 0.894, kappa 0.860, macro F1 0.824
- YASA-ear: accuracy 0.188, kappa 0.119, macro F1 0.158; consistent with a scalp-to-ear domain gap
- Scoring1 vs Scoring2: accuracy 0.938, kappa 0.917 (690 jointly valid epochs)
- Ear channel QC was not retained in that earlier log.

N1 is 20/701 epochs (2.9%); its F1 is unreliable. The BIDS `task-sleep_events.json` describes Scoring2 as manual scoring by scorer 2. The separate `task-sleep_acq-scoring_events.json` describes recording/diary event fields and has no Scoring2 field. Independence/blinding is not established in the sidecars.

Confusion matrix stage order: Wake, REM, N1, N2, N3.

| System | Confusion matrix |
|---|---|
| YASA-scalp | `[[191,2,6,0,0],[0,92,5,18,0],[2,0,11,7,0],[0,15,0,172,18],[0,0,0,0,151]]` |
| YASA-ear | `[[105,0,93,1,0],[1,0,107,7,0],[1,0,18,1,0],[4,0,194,7,0],[5,0,146,0,0]]` |
