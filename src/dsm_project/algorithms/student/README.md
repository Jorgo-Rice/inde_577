# Student-owned implementations

Write the project's course algorithms in this package. Use one module per
coherent family, for example `logistic.py`, `neighbors.py`, or `trees.py`.

Do not paste a complete implementation into a notebook and do not satisfy this
boundary with a thin wrapper around scikit-learn. Trusted libraries belong in
the reference and differential-comparison path.

Every public estimator must document its mathematics and fitted attributes,
validate inputs, reject use before fitting, control randomness, and have tests.
Register it in `../registry.py` only after its minimum contract passes.
