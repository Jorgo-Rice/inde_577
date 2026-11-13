# ADR 0002: Waze churn as the project dataset

- Status: accepted; the instructor approved the Waze dataset, including the
  fact that it is synthetic
- Date: 2026-11-13
- Supersedes: the "stand-in dataset" part of ADR 0001

## Context

ADR 0001 built the first algorithm slice on the wine data while the project
dataset was undecided. The author had already explored this synthetic Waze
user-churn dataset: exploratory analysis, engineered ratio features, and
churn models built with library decision trees, random forests, and XGBoost.
That earlier modeling compared and tuned models on its test set and did not
include its own algorithm implementations. The dataset has a real product
question, an imbalanced binary target, and enough rows (14,299 labeled) for
honest cross-validation.

## Decision

1. Use the Waze dataset for the project, building on the author's earlier
   exploratory findings and developing the modeling under this course's
   standards: student-implemented algorithms, a sealed test set, baselines,
   uncertainty, and a recorded failure analysis.
2. Load it only through `WazeChurnAdapter`, pinned by SHA-256. Keep the CSV
   out of the repository and test CI with a constructed fixture.
3. Tasks: churn classification (primary) and engagement segmentation
   (unsupervised). **Exclude regression.** Every continuous column is a usage
   measurement, so a regression target would only restate the other features.
4. Primary metrics: average precision for ranking, plus F2 at a threshold
   chosen on development data and precision/recall at fixed outreach
   budgets. Accuracy is not used, because always predicting "retained" scores
   82%.
5. Keep the wine notebook as a verification example only.

## Alternatives considered

- *Keep the wine data.* Rejected: it has no user, no decision, and is too
  small and easy to show model selection or failure analysis.
- *Choose a new public dataset.* Possible, but the Waze data already provide
  a product question and the author's earlier findings to build on.
- *Engineer a regression target* (e.g. predict next-month drives). Rejected:
  there is no next-month data.

## Consequences

- Results are about synthetic users and same-month disengagement. The product
  contract states this limitation.
- Reproducing the notebooks requires downloading the CSV. CI covers the code
  paths with the fixture but cannot reproduce the numbers.
- Revisit if committing the CSV becomes desirable, so that CI can run the
  real-data notebooks.
