# Stage 4 verification — October 15, 2026

Fresh Windows environment, Python 3.12.14. Exact versions are recorded in requirements-dev.lock.txt. pip check passed; 15 pytest tests passed; ruff and Git whitespace checks passed. The original script/README/notebook fingerprints are checked by a preservation test.

The 480-row offline fixture completed, including all four LightGBM/constant-fallback tasks, baselines, predictions, shared OLS and provenance. Two repeated test runs produced identical metrics, simulation CSV bytes and prediction CSV bytes. Nine sensitivity runs (three seeds by three curvature assumptions) completed. Every saved run's input/output fingerprints and current housing_model.py hash were checked. The fixture notebook executed in a clean kernel and matched saved OLS results.

Explicit UCI acquisition also completed: 99,826 source rows; 98,438 after cleaning and exact retained-record deduplication; 77,200 training rows and 21,238 held-out rows. The source timestamp, input hash and numerical summary are in validated-uci-summary.json. This is a new snapshot/run, not a reproduction of a historical 80,000-row claim. The UCI notebook also executes against the same saved simulation and OLS results.

Observed-rent RMSE is $589.26 versus $671.92 for the mean baseline in this one fixed held-out-city split. Time, views and rented labels are generated; their scores describe model recovery only. Views RMSE (24.0209) is almost identical to the mean baseline (24.0241). Rented AUC is 0.9504, but accuracy (0.99209) is slightly below the class-prior baseline (0.99233), illustrating rare positives and the limitations of a 0.5 threshold. No threshold tuning on this test set was performed. Do not present these synthetic scores as evidence of real rental success.

UCI's mixed-type CSV produced a nonfatal DtypeWarning; numeric coercion and explicit schema/finite/unit filters handled it. Jupyter emitted a nonfatal local TCP transport warning; notebook execution and kernel shutdown succeeded. No network calls occur in the fixture CI commands. Ubuntu-hosted CI is pending until push; Windows validation does not prove cross-platform bit-identical model results.
