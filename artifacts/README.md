# Generated artifacts

Training commands write local model bundles beneath this directory. Generated
artifacts are ignored by Git because an artifact must be rebuilt or promoted
through a release workflow rather than silently committed from one laptop.

Each model bundle should include parameters, preprocessing, feature order,
task semantics, source revision, data identity, configuration, metrics, and
software versions.
