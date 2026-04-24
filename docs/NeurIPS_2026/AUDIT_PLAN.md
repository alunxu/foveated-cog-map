# Implementation audit plan

Goal: find bugs like `deterministic=False` before reviewers do.

## Categories

| # | Category | Files | Risk | Representative bugs |
|---|----------|-------|------|---------------------|
| A | Data collection protocol | `scripts/probing/collect.py`, `scripts/eval/*.py` | HIGH (past bug) | Inconsistent `deterministic=`, wrong ckpt loaded, sensor masking logic, action-space mismatch |
| B | Probe methodology | `src/utils/probing.py`, `scripts/probing/analyze.py` | HIGH | R² clipping hides failures, α-sensitivity, episode-level split leakage, StandardScaler mis-applied |
| C | Target definitions | `analyze.py`, `goal_vector_probe.py`, `masked_heading_probe.py` | HIGH | GPS world vs episodic, compass radians convention, sin/cos encoding, ego-frame transform |
| D | Hidden-state extraction | `collect.py` `h_all = new_rnn_hidden[0, 0::2]` | MED | h vs c indices wrong, top-layer confusion, gaze-hook side effect |
| E | Custom policy impls | `src/habitat/wijmans_policy.py`, `foveated_policy.py`, `foveated_learned_policy.py`, `torch_foveation.py` | MED-HIGH | Foveation formula wrong, gaze parameter applied wrong, blind-forcing bug, NaN-sanitise edge cases |
| F | Cross-condition analyses | `analyze_cross.py`, `unaligned_cka.py`, `extended_lag_probe.py` | MED | Common-sample truncation done wrong, CKA normalisation, lag-k index off-by-one |
| G | Behavioural interventions | `scripts/eval/shortcut.py`, `transplant.py` | MED | Reset vs persistent state handling, transplant midpoint off-by-one, SPL scoring |
| H | Configs | `habitat_configs/*.yaml` | LOW-MED | `max_episode_steps` drift, `success_distance` drift, different `total_num_steps`, `num_environments` affects stats |

## Execution order (by risk × likelihood-of-finding)

1. **Category A + B + C combined** — same scripts, shared risk, likely place to find 2nd bug
2. **Category E** — custom code, hand-written, most likely original bugs
3. **Category G** — behavioural, orthogonal to probe bug but cited in paper
4. **Category F** — correlations across the two
5. **Category D + H** — mostly mechanical, low risk

## Output

For each finding:
- **Severity**: BLOCKER / MAJOR / MINOR / OK
- **Evidence**: file:line + exact issue
- **Fix**: code change + whether it requires re-running experiments

Log to `docs/NeurIPS_2026/AUDIT_FINDINGS.md`. Critical issues also
flagged here in plain text for visibility.
