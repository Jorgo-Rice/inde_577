# Data boundary

`data/local/` is ignored by Git and is the default location for student-supplied
files. Do not weaken this rule to make a notebook convenient.

The untouched template uses the Wine Recognition dataset packaged with
scikit-learn. The original data are credited to the UCI Machine Learning
Repository and described in scikit-learn's dataset documentation. The template
loads them without network access; it does not redistribute a second raw copy.

`data/fixtures/` is reserved for small, public, CI-safe records. A project fixture
must preserve the relevant schema and failure cases without exposing private or
restricted observations. Document whether a fixture is sampled, deidentified,
aggregated, or constructed.

Every project must complete `docs/DATASET_CARD.md` and provide a repeatable
ingestion or authorized-access procedure.
