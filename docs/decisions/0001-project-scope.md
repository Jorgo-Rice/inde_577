# ADR 0001: Project scope

- Status: proposed (for review with the instructor)
- Date: 2026-11-13

## Context

The final project must combine tested, student-authored implementations of
course algorithms with self-contained case-study notebooks on a chosen
dataset, inside the course template's production shell. The final dataset has
not been chosen yet, but the algorithm, testing, and notebook standards can be
built and reviewed now.

## Decision

1. Build the first vertical slice (milestone 2A) on the UCI wine data, which
   the template already loads offline. It is a **stand-in** for development and
   review, not the final project dataset.
2. Implement two classifiers from first principles under
   `src/dsm_project/algorithms/student/`: multinomial logistic (softmax)
   regression and k-nearest neighbors. They make different assumptions (linear
   class boundaries versus local similarity), so comparing them answers a real
   modeling question.
3. Hold every student estimator to the same verification standard: analytic
   cases, gradient checks where an objective exists, failure-mode tests, and
   differential tests against scikit-learn with matching objectives.
4. Draw the train/test boundary in one package function
   (`split_supervised_task`) so notebooks select models on development data
   only and open the sealed test set once.

## Alternatives considered

- *Wait for the final dataset before writing code.* Rejected: the algorithms
  and test standards do not depend on the domain, and early review lowers risk.
- *Start with regression.* The template already provides a linear-regression
  exemplar; starting with classification adds new course families.

## Consequences

- The notebook must be re-run on the project dataset once it is chosen; its
  structure is designed to carry over unchanged.
- Revisit this decision when the dataset is chosen: some algorithm families
  may become inapplicable or newly relevant.
