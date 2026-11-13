# Product contract

Replace instructional text with project-specific claims before the proposal
milestone.

## Users and decisions

Identify the intended users, the investigations or decisions the product
supports, and decisions it must not automate.

## Dataset suite

For each dataset, record its source, license, observational unit, sampling
process, time boundary, feature availability, target or unsupervised question,
and known representativeness limits.

## Task-algorithm matrix

| Course family | Project task | Applicable? | Mathematical justification | Evidence |
| --- | --- | --- | --- | --- |
| Linear regression | (stand-in) alcohol from other chemistry | Pending dataset | Template exemplar `BatchLinearRegressor` exists | — |
| Logistic regression | Cultivar classification | Yes | Tests whether classes are linearly separable; gives class probabilities | `SoftmaxRegression`, notebook 01 |
| Nearest neighbors | Cultivar classification | Yes | Tests whether local chemical similarity implies the same class | `KNearestNeighborsClassifier`, notebook 01 |
| Support vector machines and kernels | | Pending dataset | | |
| Decision trees | | Pending dataset | | |
| Ensembles | | Pending dataset | | |
| Neural networks | | Pending dataset | | |
| Clustering | | Pending dataset | | |
| Dimensionality reduction | | Pending dataset | | |

Current rows describe the stand-in wine slice (see ADR 0001) and will be
revised for the project dataset.

Mark exclusions explicitly. Forcing an algorithm onto an incompatible target
does not demonstrate breadth.

## Evaluation contract

Define development and final-test roles, metrics, uncertainty evidence,
baselines, selection rules, subgroup or condition slices, and failure thresholds
before inspecting final-test results.

## System interfaces

Define the data boundary, training command, experiment artifact, prediction
API, user interface, deployment environment, logs, metrics, alerts, and
rollback path.

## Responsible-use boundary

Describe privacy, security, fairness, misuse, uncertainty, and human-oversight
requirements. State what a successful technical demonstration does not prove.
