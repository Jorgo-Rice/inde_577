# Project checklist

## Required replacements

- [ ] Replace the template title and description with the product identity.
- [ ] Identify intended users, supported decisions, and prohibited uses.
- [ ] Complete the dataset card and product contract.
- [ ] Replace the example dataset configuration with project data adapters.
- [ ] Define at least one regression, one classification, and one unsupervised
      or representation-learning task when the domain supports them.
- [ ] Explain every algorithm exclusion or incompatibility.
- [ ] Define leakage-resistant development and final-evaluation boundaries.
- [ ] Record individual and shared team ownership.

## Required implementation

- [ ] Course algorithms are student-authored under
      `src/dsm_project/algorithms/student/` and entered through the registry.
- [ ] Each student algorithm is compared against an analytic result, a trusted
      reference implementation, or both; agreement is tested, not assumed.
- [ ] Reusable data, feature, experiment, evaluation, and service code is in
      `src/dsm_project/`.
- [ ] Every applicable course algorithm runs through a documented interface.
- [ ] Trusted-library references are visibly distinguished from student code
      in source, results, notebooks, and the final presentation.
- [ ] Training emits a versioned artifact with configuration and provenance.
- [ ] The API validates inputs and reports model identity.
- [ ] The user-facing interface answers a defined user need.
- [ ] Structured logs and operational metrics support failure diagnosis.

## Required evidence

- [ ] Every project notebook tells one coherent, self-contained story about an
      algorithm or justified mixture of algorithms on the chosen data.
- [ ] Every **STUDENT WORK** prompt from the guided template has been replaced
      with domain-specific reasoning, code, evidence, and conclusions.
- [ ] Student notebooks under `notebooks/` follow the case-study standard and
      call reusable project-package code on the configured dataset.
- [ ] Each method report includes purposeful exploration, mathematical
      exposition, a baseline or trusted comparison, diagnostics, and limits.
- [ ] Unit, property, differential, integration, API, and system tests cover
      relevant risks.
- [ ] Example and project notebooks restart and run from top to bottom.
- [ ] CI passes from a clean clone.
- [ ] Model-selection evidence retains losing candidates and baselines.
- [ ] Final-test evidence is produced only after selection is frozen.
- [ ] Limitations, responsible use, rollback, and incident response are written.
- [ ] The final release identifies source, data, configuration, and model versions.
