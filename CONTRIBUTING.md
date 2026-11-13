# Contributing

Use one issue and one branch for one coherent change. A pull request must state
the user or scientific need, mathematical contract, files changed, verification
evidence, data implications, and known limitations.

Before requesting review, run:

```bash
uv run ruff check src tests scripts
uv run pytest -q
uv build
```

Reviewers should examine scientific meaning, data leakage, mathematical
correctness, interface behavior, failure cases, reproducibility, security, and
operational consequences—not merely whether the code runs.

Never include grades, student IDs, accommodations, credentials, private data,
or restricted research material in issues, commits, logs, or pull requests.

## Student algorithm changes

An algorithm pull request must add or update code under
`src/dsm_project/algorithms/student/`, register the implementation explicitly,
and add tests. A notebook-only implementation is incomplete. A wrapper that
only instantiates a scikit-learn estimator is a reference adapter, not a
student-authored algorithm.
