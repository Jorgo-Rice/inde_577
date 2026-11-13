# Algorithm boundary

This is a student implementation boundary, not a catalog of finished course
algorithms. The template supplies simple baselines and one transparent
batch-gradient linear regressor to demonstrate the expected interface,
validation, documentation, fitted state, and tests. Students build the
remaining applicable algorithms in `student/` and connect them to the system
through `registry.py`.

The scikit-learn estimators under `experiments/` are trusted references. Use
them for differential tests and empirical comparisons; do not relabel them as
student implementations.

Every new estimator needs:

- a mathematical specification;
- typed, documented public behavior;
- ordinary, boundary, and invalid-input tests;
- analytically checkable cases;
- a trusted differential comparison when one exists;
- reproducible randomness;
- numerical and computational limitations; and
- a worked example using the project data boundary.
