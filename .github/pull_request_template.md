## Purpose

<!-- What user, scientific, or operational problem does this solve? -->

## Contract

<!-- State task meaning, interfaces, assumptions, and expected failures. -->

## What changed

<!-- Summarize the smallest coherent change. -->

## Evidence

- [ ] Targeted tests pass.
- [ ] `uv run ruff check src tests scripts` passes.
- [ ] `uv run pytest -q` passes, or omissions are explained.
- [ ] Changed notebooks restart and run from top to bottom.
- [ ] Data provenance, units, schema, and split boundaries are documented.
- [ ] Reusable behavior is in `src/`, not only notebook state.
- [ ] No secrets, credentials, private data, or restricted data are included.
- [ ] Operational, compatibility, or artifact consequences are documented.

## Reviewer focus

<!-- Name mathematical, data, security, reliability, or integration risks. -->

## Related issue

<!-- Use "Closes #123" when merging should close an issue. -->
