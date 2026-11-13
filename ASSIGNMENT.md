# Final project assignment

## The outcome

Build one coherent data-science or machine-learning product around a dataset
chosen by your team. The finished repository must show how a scientific or
operational question becomes mathematics, tested package code, reproducible
evidence, a usable system, and an observable release.

This is not a collection of disconnected homework solutions. Every algorithm
must contribute to the same project story: it answers a relevant question,
provides a meaningful baseline, tests an alternative modeling assumption, or
reveals useful structure in the data.

## What you will write

Your work has two connected forms.

### 1. Reusable software

Implement applicable course algorithms under
`src/dsm_project/algorithms/student/`. Put reusable ingestion, validation,
feature, evaluation, artifact, service, and monitoring behavior under `src/`
or `scripts/`. Test that behavior under `tests/`.

### 2. Guided applied case-study notebooks

Write a sequence of self-contained notebooks under `notebooks/`. Each notebook
must use machine learning and data analysis to answer a real scientific,
engineering, social, or operational question. It should read as a technical
case study that a new teammate can follow without hearing the lecture that
produced it.

A notebook may use:

- one algorithm in depth to answer the question;
- two or more algorithms whose comparison answers the question; or
- a coherent pipeline, such as representation learning followed by clustering
  or classification.

The applied question—not the algorithm—is the organizing unit. Do not combine
unrelated questions merely to reduce the notebook count, and do not create a
notebook solely because an algorithm appeared in the course. If a method is
scientifically incompatible with the data, document and defend the exclusion.

The course examples are completed demonstrations of the reasoning pattern.
They are not student deliverables and are not intended to be copied with only
the dataset name changed. Start from `notebooks/00-guided-case-study-template.ipynb`
and make the investigation your own.

## The required story in every applied notebook

Every notebook must contain enough explanation, code, and evidence to answer
the following questions in order.

1. **Question and decision.** Who might use the result? What scientific or
   operational question is being asked? What would a useful answer change?
2. **Data contract.** What does one observation represent? Where did the data
   come from? Which information is available at prediction or analysis time?
3. **Exploration with a purpose.** Which summaries and plots test assumptions,
   reveal scale, imbalance, dependence, missingness, or possible leakage?
4. **Mathematical model.** What are the inputs, fitted quantities, objective or
   update rule, outputs, assumptions, and important hyperparameters?
5. **Student implementation.** Which class or function in the project package
   implements the method? What interface and invariants does it promise?
6. **Verification.** Which analytic, hand-computable, property, and trusted-
   library comparisons show that the implementation behaves as intended?
7. **Evaluation design.** Which data are used for fitting, tuning, and final
   evaluation? Why are the metrics and baselines appropriate to the question?
8. **Results and uncertainty.** What did the experiment show, how variable is
   the evidence, and what is the practical—not merely numerical—meaning?
9. **Failure analysis.** Where does the method fail? Examine residuals,
   confusion patterns, subgroups, instability, outliers, cost, or drift risks.
10. **Judgment.** What can be concluded, what cannot, and what should be tried
    or changed next? Should this method be promoted into the product?

Each figure must answer a stated question, label variables and units, and be
interpreted in prose. Each displayed metric must have a baseline and context.

## What the notebook must not hide

The notebook orchestrates an experiment and explains its meaning. It must not
be the only place where the estimator, data transformation, metric, API, or
monitoring logic exists. If code is reusable, testable, or needed by the
product, move it into the package and import it.

The notebook must identify student-authored code and trusted-library code
explicitly. Calling a library estimator is useful reference evidence, but it
does not satisfy an implementation requirement.

## Repository evidence

The final repository must include:

- a completed product contract, dataset card, model card, and operations plan;
- lawful access to the chosen data and a CI-safe test fixture;
- tested student implementations of applicable course algorithms;
- self-contained case-study notebooks that run from clean kernels;
- baselines, retained candidate results, uncertainty, and failure analysis;
- a versioned model artifact and documented promotion decision;
- a tested API and user-facing behavior that serve the declared user;
- structured logs, monitoring evidence, rollback instructions, and an incident
  exercise;
- issues, branches, pull requests, reviews, and passing CI that show how the
  team built the product; and
- a tagged final release whose source, configuration, data provenance, and
  model identity can be recovered.

## A useful minimum sequence

Most projects will need the following narrative sequence, though the exact
number of method notebooks depends on the data and justified scope.

1. Data, question, and baseline.
2. Supervised method case studies that match the available targets.
3. Unsupervised or representation case studies that answer a real structural
   question.
4. Model comparison, selection, and error analysis.
5. Production, monitoring, and release evidence.

Use [`MILESTONES.md`](MILESTONES.md) for timing and
[`PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md) for the final audit. The central
standard is coherence: the question, mathematics, implementation, tests,
evidence, product behavior, and claims must agree.
