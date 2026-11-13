# Notebook case-study standard

Project notebooks are self-contained, executable technical case studies backed
by the project package. They should combine the accessibility of a strong
public data-science article with the evidence and restraint expected in a
professional analysis. The student supplies the decisions and intellectual
work; the template supplies a sequence of questions that prevents important
professional steps from disappearing.

A notebook uses data analysis and machine learning to answer a real-world or
scientific question. It may use one algorithm or a justified mixture of
algorithms. The unit of organization is the investigation, not the syllabus:
methods belong together only when their comparison or composition answers one
governing question.

## Reader contract

Assume the reader knows basic Python and course mathematics but has not heard
the lecture, seen the data, or watched the team work. Define notation before
using it. Introduce code by explaining what question it answers. Interpret the
output immediately afterward. Make transitions explicit so the report reads as
an argument rather than a sequence of cells.

Prompts in the guided template are minimum obligations, not requested phrases.
Replace them with domain-specific prose and analysis; do not submit the prompts
themselves.

## Required narrative arc

Every substantial notebook should let a reader answer these questions in order:

1. **What is the question?** Name the user, decision or scientific question,
   observation unit, target or structure of interest, and why the result matters.
2. **Where did the data come from?** Link the source, state the license and
   retrieval date, and load it through `dsm_project.data` rather than an
   unexplained notebook path.
3. **Can the data support the question?** Audit shape, types, units, missingness,
   duplicates, dependence, sampling, class balance, and information available
   at use time.
4. **What does the data look like?** Use purposeful tables and plots. Each visual
   needs a question, labeled axes and units, a caption or interpretation, and an
   explanation of what it does not establish.
5. **What is the method?** Define notation, objective, assumptions, fitted
   quantities, prediction rule, hyperparameters, and failure modes before the
   code call.
6. **Where is the implementation?** Import reusable student code from
   `dsm_project`. Do not define the production estimator, data adapter, metric,
   or serving logic only in a notebook.
7. **How was it evaluated?** State split or resampling boundaries, baselines,
   metrics, uncertainty, slices, and whether the evidence is development,
   validation, or final-test evidence.
8. **What failed?** Inspect residuals, confusion patterns, unstable clusters,
   outliers, subgroups or conditions, computational limits, and plausible data
   or concept drift.
9. **What follows?** Give a bounded conclusion, limitations, responsible-use
   statement, next experiment, and production consequence.

The notebook must also identify the student-authored package symbols and tests
that support the analysis. A result without an inspectable implementation and
verification path is incomplete evidence.

## Code boundary

Good notebook code orchestrates and communicates:

```python
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.experiments import evaluate_supervised_estimator
```

Package code performs reusable work. Notebook code may create a diagnostic plot
or a one-time explanatory calculation, but if another notebook, a test, the CLI,
or the API needs the behavior, move it under `src/dsm_project/` and test it.

## Visual and statistical standard

- Display a compact preview; never dump an entire dataset.
- Explain why each statistic or plot is relevant to the question.
- Show distributions before relying on averages.
- Label units and categories and use readable legends and color choices.
- Separate exploratory claims from confirmatory or final-test claims.
- Prefer effect sizes, uncertainty and error structure to a single score.
- Compare with a simple baseline and a trusted implementation.
- Preserve failed candidates and surprising results when they teach something.
- Do not interpret correlation, clusters, feature importance, or a projection as
  causation.

## Reproducibility standard

A reviewer must be able to restart the kernel and run all cells from top to
bottom using the named project environment. The notebook must not depend on
hidden state, a private absolute path, a manual download that is undocumented,
or a live secret embedded in a cell. Clear outputs before a pull request unless
the course submission policy explicitly asks for rendered outputs.

## Review questions

Reviewers should ask:

- Does the narrative match the package behavior?
- Are notation, units, labels, and data boundaries defined before use?
- Could any transformation see validation or test information too early?
- Are reference-library and student-authored results visibly distinguished?
- Do the diagnostics test the model's likely failure modes?
- Would a new teammate know what to reproduce, challenge, or build next?
