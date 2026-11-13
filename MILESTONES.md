# Recommended milestones

## 0. Repository and team contract

Verify the untouched template, record participants, enable branch protection,
and agree on review and maintenance responsibilities.

## 1. Product and data contract

Define the user, decision, observational unit, provenance, data access,
information boundary, tasks, exclusions, and responsible-use boundary.

## 2. Data boundary and baselines

Implement ingestion, schema validation, a CI-safe fixture, reproducible
partitions, and simple baselines before adding complex models.

## 2A. Student implementation boundary

Implement one transparent estimator completely, register it, verify it on an
analytic case, compare it with a trusted reference, and drive it through the
same configured experiment path. Use this reviewed vertical slice as the
standard for subsequent algorithms.

## 3. Supervised experiments

Implement and run applicable regression and classification families. Preserve
preprocessing inside fitted pipelines, compare against trusted references, and
retain complete candidate evidence.

## 4. Unsupervised and representation experiments

Define what structure means without labels. Evaluate stability, geometry,
reconstruction, density, or scientific interpretation as appropriate. Labels,
when available, are audit evidence rather than hidden fitting targets.

## 5. Selection and artifact promotion

Freeze metrics and selection rules, compare candidates, open the sealed test
set once, and package the approved model with provenance and configuration.

## 6. End-to-end product

Connect the durable data boundary, model artifact, backend API, user-facing
interface, and deployment artifact. Test the complete path.

## 7. Operations and final review

Exercise logs, metrics, monitoring, failure response, rollback, and delayed
quality evidence. Produce a tagged release and complete the oral-defense record.
