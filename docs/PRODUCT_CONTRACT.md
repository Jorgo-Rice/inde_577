# Product contract: Waze churn risk

## Users and decisions

**Intended user.** A Waze product manager planning retention outreach (in-app
prompts or emails).

**Supported decisions.**

- Which current users to prioritize for a retention message, given an
  outreach budget or a recall target.
- Which behaviors are associated with churn risk, to inform hypotheses for
  product experiments.
- Which behavioral segments exist and how their churn rates differ
  (unsupervised task, planned).

**Decisions it must not automate.**

- Any action that changes a user's service, price, or account.
- Claims that a behavior *causes* churn. The data are observational and
  synthetic.
- Early-warning claims. Activity and churn are measured in the same month
  (see the dataset card), so the score detects disengagement already under
  way rather than forecasting it.

## Dataset suite

| Dataset | Source and license | Observation unit | Time boundary | Target or question |
| --- | --- | --- | --- | --- |
| Waze user churn (synthetic) | Waze/Google synthetic data distributed on Kaggle; license to be confirmed, so not committed | One user | One month of activity plus lifetime totals | `churned` this month; engagement segments |
| Wine recognition (verification only) | UCI via scikit-learn, CC BY 4.0 | One wine sample | None | Cultivar; used only to verify algorithms on clean data |

Details are in [`DATASET_CARD.md`](DATASET_CARD.md).

## Task-algorithm matrix

| Course family | Project task | Applicable? | Mathematical justification | Evidence |
| --- | --- | --- | --- | --- |
| Linear regression | — | Excluded | Every continuous candidate target (driving days, km driven, sessions) is a usage measurement that restates the other features, so predicting it answers no user question. The template exemplar remains as verified code. | ADR 0002 |
| Logistic regression | Churn risk | Yes | Churn falls monotonically with activity and tenure; a linear log-odds model is a transparent baseline with readable coefficients | `SoftmaxRegression`, notebook 02 |
| Nearest neighbors | Churn risk | Yes | Tests whether *similar usage* implies similar risk without a linearity assumption | `KNearestNeighborsClassifier`, notebook 02 |
| Decision trees | Churn risk | Yes | Can capture thresholds and interactions such as tenure × activity; the data turned out to be close to additive, so the tree is kept for readable rules, not promoted | `ClassificationTree`, notebook 03 |
| Ensembles | Churn risk | Yes (next) | Bagging and boosting reduce single-tree variance; the earlier project's XGBoost becomes a trusted reference | Planned (lecture 15) |
| Support vector machines and kernels | Churn risk | Optional | Applicable, but kernel SVMs scale poorly to 10k+ rows and give no calibrated probabilities without extra fitting | Decide after trees |
| Neural networks | Churn risk | Optional | Applicable; only worth adding if trees show non-linear signal that linear models miss | Decide after trees |
| Clustering | Engagement segments | Yes (planned) | k-means on standardized usage to find segments; churn labels used only to audit segments | Planned |
| Dimensionality reduction | Engagement segments | Yes (planned) | PCA to visualize and compress correlated usage counts before clustering | Planned |

## Evaluation contract

- **Partition.** One stratified 75/25 split (seed 438): 10,724 development
  users and 3,575 sealed test users. The test set is scored once per
  frozen candidate set.
- **Selection.** 5-fold stratified cross-validation on development users;
  out-of-fold probabilities are retained for every candidate.
- **Ranking metric.** Average precision (baseline = churn rate, about 0.177).
- **Decision metric.** F2 at a threshold chosen on out-of-fold probabilities,
  plus precision and recall at fixed outreach budgets (top 10%, 20%, 30%).
- **Baselines.** Majority class ("always retained"); later models must beat
  the student softmax model (development AP about 0.39).
- **Uncertainty.** Bootstrap intervals on test metrics.
- **Slices.** Recall by activity level and by tenure; device checked for
  disparity.
- **Selection rule.** Highest development AP; differences under 0.005 are
  ties, broken toward the simpler model.

## System interfaces

| Interface | Current state |
| --- | --- |
| Data boundary | `WazeChurnAdapter` (`configs/waze.toml`), checksum-verified |
| Training | Notebook 02 for now; promotion to `scripts/train.py` planned |
| Artifact | Template bundle format; Waze model not yet promoted |
| API | Template FastAPI service; churn-risk endpoint planned |
| Monitoring | Template structured logs; input-range checks planned |
| Rollback | Previous artifact version (planned with promotion) |

## Responsible-use boundary

- Scores are risk *rankings* for prioritizing outreach, reviewed by a person;
  they are not judgments about individual users.
- The data are synthetic. Nothing here is evidence about real Waze users.
- No personal identifiers are used; the user ID is an arbitrary row number.
- Device type shows no churn difference and is retained only as a feature;
  model behavior by device will be checked before promotion.
- A good test score does not show that outreach *reduces* churn. That needs
  an experiment such as an A/B test.
