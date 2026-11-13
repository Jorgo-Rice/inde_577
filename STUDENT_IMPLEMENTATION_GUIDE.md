# Student implementation guide

## What the template gives you

The untouched repository is a thin, working production system: configuration,
a validated data contract, reproducible reference experiments, artifact
serialization, a prediction API, a minimal client, structured logs, tests,
containers, and continuous integration. Run it first so that you know what a
healthy end-to-end path looks like.

The shell is intentionally real. Your work should strengthen it rather than
replace it with disconnected notebook code.

## What you must write

Your team owns four kinds of implementation:

1. **Scientific code:** course algorithms appropriate to the project, written
   under `src/dsm_project/algorithms/student/`.
2. **Domain code:** ingestion, validation, features, task definitions, and
   leakage-resistant partitions for your data.
3. **Evidence code:** metrics, comparisons, stress tests, and model-selection
   logic that match the actual decision.
4. **Product code:** the API, client behavior, monitoring, and failure handling
   needed by your declared user.

Notebooks may demonstrate all four, but they may not be their only home.

## The implementation loop

For every applicable algorithm:

1. Write its mathematical contract: inputs, fitted quantities, objective,
   update rule or solver, outputs, assumptions, and failure conditions.
2. Implement a small public interface with `fit` and `predict`, `transform`, or
   another task-appropriate operation.
3. Add the implementation to `STUDENT_ALGORITHMS` in
   `src/dsm_project/algorithms/registry.py`.
4. Test an exactly or analytically checkable case.
5. Test invalid shapes, invalid hyperparameters, use before fitting, and
   reproducibility where randomness is present.
6. Compare with a trusted library on data for which the objectives and
   conventions agree. Explain any expected disagreement.
7. Run the estimator through the configured experiment and data boundaries.
8. Use a notebook to communicate the mathematics, visualization, and evidence.
9. Open a focused pull request and have a teammate review both the science and
   software behavior.

## Course-topic implementation map

| Family | Student-owned core | Minimum comparison evidence |
| --- | --- | --- |
| Linear regression | prediction, loss, gradient, optimizer, fitted state | analytic line and trusted least-squares solver |
| Logistic regression | sigmoid/logit, stable log loss, gradient, probabilities | finite-difference gradient and trusted classifier |
| Linear SVM | margin score, hinge objective, regularized update | separable geometry and trusted linear SVM |
| Nearest neighbors | distance, neighbor search, voting/aggregation | hand-computable neighborhood and trusted estimator |
| Decision tree | split criterion, split search, stopping, traversal | hand-computable split and trusted shallow tree |
| Ensembles | bootstrap/subsample logic, aggregation or boosting update | single-model baseline and trusted ensemble |
| Neural network | layers, activations, forward pass, backpropagation, update | gradient check and trusted/autodiff reference |
| Clustering | assignment, prototype/update rule, stopping | hand-computable clusters, stability, trusted method |
| Dimensionality reduction | centering, fitted representation, transform | reconstruction/variance identity and trusted method |

The domain may make a family inapplicable. A rigorous incompatibility argument
is better than forcing meaningless code onto the data, but it must be recorded
in the product contract and approved through review.

## Trusted references are measuring instruments

Scikit-learn is present because a professional engineer verifies new code
against mature tools. The reference catalog keeps the template operable before
student estimators exist and provides differential comparisons afterward.

A call to `fit_supervised_candidate(...)` uses a trusted reference. A class
registered in `STUDENT_ALGORITHMS` is student-owned. Results, plots, logs, and
presentations must preserve that distinction.

## Promotion into the product

The example API initially serves a reference bundle to prove the deployment
path. After an implementation passes its mathematical, software, integration,
and review gates, change the training path to promote that registered student
estimator. Record the decision, artifact version, rollback target, and known
limitations. That promotion is the moment an experiment becomes product
behavior.
