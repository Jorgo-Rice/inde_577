# Team ownership and integration

## Participants

Solo project: Jorgo Qirjaj (@Jorgo-Rice).

## Shared core

Not applicable while the project is solo. Course-template infrastructure
(configuration, data adapters, experiment runners, artifact, API, CI) is
inherited from the template and credited to the course.

## Individual vertical slices

| Slice | Owned interfaces | Tests | Notebook |
| --- | --- | --- | --- |
| Classification algorithms | `SoftmaxRegression`, `KNearestNeighborsClassifier`, `cross_entropy_objective`, `pairwise_distances` | `tests/test_student_algorithms.py` | `notebooks/01-wine-cultivar-softmax-vs-knn.ipynb` |
| Evaluation boundary | `split_supervised_task`, `bootstrap_metric_interval`, `choose_threshold`, `ThresholdedClassifier` | `tests/test_evaluation_and_split.py`, `tests/test_thresholds.py` | notebooks 01, 02 |
| Waze data boundary | `WazeChurnAdapter`, `validate_waze_frame`, `prepare_waze_observations` | `tests/test_waze_data.py` | notebook 02 |

## Integration and conflict history

- The template baselines (`MeanRegressor`, `MajorityClassifier`) could not be
  cloned by scikit-learn cross-validation. They now inherit the standard
  scikit-learn estimator bases so baselines go through the same selection
  procedure as every candidate.
- The holdout split was moved out of `evaluate_supervised_estimator` into
  `split_supervised_task` so that cross-validation and final evaluation provably
  use the same development/test boundary.

- Differential testing showed that binary softmax regression matches
  scikit-learn's logistic regression at `C = 2/(n*lambda)`, not
  `1/(n*lambda)`, because softmax splits one weight vector across two
  classes. The derivation is recorded in `logistic.py` and notebook 02.

## Maintenance responsibility

Jorgo Qirjaj maintains all components through final review.
