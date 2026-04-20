# Upstream PR materials: habitat-baselines NaN sanitisation

This directory holds the draft materials for contributing our NaN-sanitisation fix back to `facebookresearch/habitat-lab`. Nothing here has been submitted yet — review before opening the issue / PR.

## Files

- **`ISSUE.md`** — the GitHub issue text describing the bug, the root cause, the observed symptoms with our actual data, and the proposed fix.
- **`patch.diff`** — the concrete code change against `habitat-baselines/habitat_baselines/rl/ppo/ppo.py`. Equivalent to the monkey-patch in `src/habitat/wijmans_policy.py` but native (no runtime patching needed).

## How to submit (when ready)

1. Read through `ISSUE.md`, adjust any phrasing.
2. File an issue at https://github.com/facebookresearch/habitat-lab/issues with the content of `ISSUE.md`.
3. Wait for maintainer feedback (typically 2–7 days). They may prefer a different API shape (e.g. a flag to disable, a different metric name).
4. Fork `habitat-lab`, create a branch, apply `patch.diff`, push.
5. Open the PR linked to the issue; include a short test (our `tests/test_nan_sanitisation.py` can be adapted to test the native implementation).

## Why we kept a runtime monkey-patch in our own code

Even if upstream accepts the PR, users on older habitat-baselines versions will not benefit. Keeping the monkey-patch in `src/habitat/wijmans_policy.py` means anyone using our codebase inherits the fix regardless of which habitat-baselines commit they have pinned.

Once the upstream fix lands and we bump our habitat-baselines dep to a version that includes it, we can remove the runtime patch.
