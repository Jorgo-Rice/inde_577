# Student project case studies

This directory holds the applied scientific narrative of the project. Begin by reading
[`ASSIGNMENT.md`](../ASSIGNMENT.md), then duplicate
`00-guided-case-study-template.ipynb` for each coherent investigation.

## What “one notebook” means

A notebook is not automatically one algorithm and it is not automatically one
course week. It is one complete, readable investigation that uses data
analysis and machine learning to answer one real-world or scientific question.
It may:

- teach and evaluate one algorithm in depth;
- compare several algorithms that answer the same question; or
- compose methods into a justified pipeline.

For example, a regression notebook might compare a student linear model with a
tree ensemble because the scientific question is whether nonlinear structure
materially improves prediction. A representation notebook might use PCA before
clustering because the question concerns stable low-dimensional structure.

Split the work when methods answer unrelated questions. Combine it when the
comparison itself is the story. Do not force every course algorithm onto data
for which it is meaningless; explain the incompatibility in the project scope.

## How to use the template

1. Copy `00-guided-case-study-template.ipynb` to a numbered, descriptive name,
   such as `02-linear-and-tree-regression.ipynb`.
2. Replace every **STUDENT WORK** prompt. Do not leave the example question,
   configuration, figure, metric, or interpretation unchanged.
3. Implement reusable behavior in `src/dsm_project/` and tests in `tests/`.
4. Import that behavior into the notebook through the public package API.
5. Restart the kernel and run the complete notebook from top to bottom.
6. Ask a teammate to review both the scientific argument and package code.

The notebook must stand on its own: define notation, explain why code is being
run, interpret every important result, and state limitations. A reader should
not need the lecture or commit history to understand the investigation.

Reference reports under `examples/` show completed course-topic stories. They
are models of reasoning, not submissions. Follow the full
[`notebook case-study standard`](../docs/NOTEBOOK_CASE_STUDY_STANDARD.md).
