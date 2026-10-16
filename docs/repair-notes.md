# Stage 4 housing repair

Baseline: b43f60968fb7d71d03e254e088b4e03ba8845c3e.

The original four tracked files remain recoverable in Git; original script/notebook/README are additionally archived under docs/historical. Historical requirements.txt was empty. The archived notebook is evidence of the earlier state, not an executable dependency of the repair.

Changes: import-safe explicit CSV CLI; fixed seeded simulations; explicit observed/synthetic feature sets; rent dispersion removed; held-out city groups; train-only preprocessing; shared row identities; finite/schema validation; nonempty amenity counts; count-based HHI; saved realization; baselines; nullable AUC; shared OLS; nine-run seed/curvature sensitivity; explicit source acquisition; dependency ranges/version snapshot; offline CI and clean-kernel notebook checks.

The fixture is entirely invented. Real UCI acquisition and any resulting empirical run must be reported separately. Historical conference findings and approximate resume row counts are not validated by fixture tests. HC3 is not a causal inference strategy. City labels are not neighborhoods and may conflate state geography. Same software versions/seed/input support repeatability; different platforms or libraries need not produce bit-identical fitted models.

Primary references: [UCI dataset](https://archive.ics.uci.edu/dataset/555/apartment%2Bfor%2Brent%2Bclassified), [group split semantics](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html), [LightGBM determinism parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
