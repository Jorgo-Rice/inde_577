# Team ownership and integration

## Participants

Solo project: Jorgo Qirjaj (GitHub handle to be recorded).

## Shared core

Not applicable while the project is solo. Course-template infrastructure
(configuration, data adapters, experiment runners, artifact, API, CI) is
inherited from the template and credited to the course.

## Individual vertical slices

| Slice | Owned interfaces | Tests | Notebook |
| --- | --- | --- | --- |
| Classification algorithms | `SoftmaxRegression`, `KNearestNeighborsClassifier`, `cross_entropy_objective`, `pairwise_distances` | `tests/test_student_algorithms.py` | `notebooks/01-wine-cultivar-softmax-vs-knn.ipynb` |
| Evaluation boundary | `split_supervised_task`, `bootstrap_metric_interval` | `tests/test_evaluation_and_split.py` | same |

## Integration and conflict history

- The template baselines (`MeanRegressor`, `MajorityClassifier`) could not be
  cloned by scikit-learn cross-validation. They now inherit the standard
  scikit-learn estimator bases so baselines go through the same selection
  procedure as every candidate.
- The holdout split was moved out of `evaluate_supervised_estimator` into
  `split_supervised_task` so that cross-validation and final evaluation provably
  use the same development/test boundary.

## Maintenance responsibility

Jorgo Qirjaj maintains all components through final review.
