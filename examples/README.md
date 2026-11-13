# Completed course-topic case studies

These notebooks are completed stories showing how a course method can support
a professional investigation. They are reference models, not student
submissions and not fill-in-the-blank worksheets. Students should study the
reasoning pattern, then apply it to their own question and chosen data through
the guided template under `notebooks/`.

Each report imports reusable behavior from `src/dsm_project/`, uses the same
configured data boundary, and runs in CI. Where an example calls a
scikit-learn estimator, it is showing the trusted reference path. Student
projects must make their own registered implementation appear alongside that
reference.

Each example follows the project case-study arc: motivating question, package
data boundary, focused exploration, mathematical model, package experiment,
diagnostics, limitations, and a concrete student extension. Read
[`docs/NOTEBOOK_CASE_STUDY_STANDARD.md`](../docs/NOTEBOOK_CASE_STUDY_STANDARD.md)
before creating project notebooks.

| Example | Course topic | Evidence produced |
| --- | --- | --- |
| `00-data-and-task-contract.ipynb` | Data, provenance, task, and leakage contracts | schema and task audit |
| `supervised/01-linear-regression-and-gradient-descent.ipynb` | Least squares and first-order optimization | regression metric and gradient trace |
| `supervised/02-logistic-regression-and-decisions.ipynb` | Probabilities, log loss, and decisions | classification evidence |
| `supervised/03-svm-margins-and-kernels.ipynb` | Margins and kernel geometry | linear/RBF comparison |
| `supervised/04-neighbors.ipynb` | Local geometry and scaling | neighbor-model evidence |
| `supervised/05-trees-and-ensembles.ipynb` | Recursive partitions, forests, and boosting | family comparison |
| `supervised/06-neural-networks.ipynb` | Composition and nonlinear models | neural candidate evidence |
| `supervised/07-model-selection.ipynb` | Selection as estimation | retained candidate table |
| `unsupervised/01-clustering.ipynb` | Clustering without fitting labels | silhouette and stability |
| `unsupervised/02-dimensionality-reduction.ipynb` | PCA and representation evidence | retained variance and projection |

## What students transfer—and what they do not

Transfer the narrative discipline: question, data contract, purposeful
exploration, mathematics, package implementation, verification, evaluation,
failure analysis, and bounded conclusion. Do not transfer the wine question,
plots, metric choices, conclusions, or code unchanged. Those choices must be
made again for the student's domain.

A project notebook may focus on one method or compare several methods when the
comparison answers one coherent question. Before fitting anything, state what
one row represents, what information exists at analysis or prediction time,
and why each method is compatible with the task.

The concrete student instructions live in [`ASSIGNMENT.md`](../ASSIGNMENT.md)
and the reusable starting structure is
[`notebooks/00-guided-case-study-template.ipynb`](../notebooks/00-guided-case-study-template.ipynb).
