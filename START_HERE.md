# Start here

Read [`ASSIGNMENT.md`](ASSIGNMENT.md) first. The final project is one coherent
data product supported by tested package code and self-contained case-study
notebooks—not a folder of unrelated model calls.

## 1. Create your project repository

Use **Use this template** on GitHub. Choose a descriptive repository name and
clone the new repository. Do not submit work by modifying the course template
repository itself.

## 2. Prepare the environment

Open the repository root in VS Code and run:

```bash
uv sync --locked
uv run python scripts/setup_project.py
uv run dsm-project verify
```

A successful setup verifies the package, example data, configuration, and
notebook kernel. Open a notebook and select **DSM Project**.

## 3. Verify the untouched template

```bash
uv run pytest -q
uv run ruff check src tests scripts
```

Commit your first project change only after the untouched template passes.

## 4. Add your data safely

1. Put local files under `data/local/`; Git ignores that directory.
2. Copy `configs/student.toml.example` to `configs/student.toml`.
3. Declare provenance, license, observation unit, features, targets, and split
   boundaries in `docs/DATASET_CARD.md`.
4. Implement or configure an adapter under `src/dsm_project/data/`.
5. Add a small public, deidentified, or structurally equivalent fixture that
   CI may use without credentials.
6. Add validation tests before fitting a model.

If the data cannot legally or ethically be represented by a public fixture,
test the schema and transformations with constructed records and document the
gap. Never weaken privacy controls merely to make CI convenient.

## 5. Replace the example product

The initial API uses a trusted reference model so that the untouched template
has a complete, testable vertical slice. That is scaffolding, not the finished
assignment. Read `STUDENT_IMPLEMENTATION_GUIDE.md`, implement the course
algorithms under `src/dsm_project/algorithms/student/`, add them to the explicit
registry, and promote an appropriate student implementation into the product.

The example notebooks call reusable package functions. Preserve that
direction: notebooks may orchestrate, visualize, and explain, but product
behavior belongs under `src/`. Do not copy large notebook cells into production
modules without first designing and testing their interfaces.

Create project reports under `notebooks/` from
`notebooks/00-guided-case-study-template.ipynb` and follow
`docs/NOTEBOOK_CASE_STUDY_STANDARD.md`. Each report is one complete
investigation of an algorithm or justified algorithm mixture on the team's
chosen data. It should teach the method, use student package code, compare with
a clearly labeled trusted reference, diagnose errors, and state limitations.

## 6. Work through reviewed branches

```bash
git switch -c team/short-purpose
git add PATHS-YOU-INTEND-TO-COMMIT
git commit -m "Describe the completed behavior"
git push -u origin team/short-purpose
```

Open a pull request, complete the evidence checklist, request review, and wait
for CI before merging.
