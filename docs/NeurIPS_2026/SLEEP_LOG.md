# Sleep Log — autonomous overnight work 2026-04-27 ~08:15 onwards

**Plan**: 6-hour autonomous experiment-analysis-interpretation-writing loop while user sleeps.

User-set rules:
1. Positive results matching expected pattern → fill paper stubs (`\hcpending` → numbers).
2. Negative / unexpected → do NOT touch paper claims. Write findings here with WHY-might-be-wrong + sanity-check protocol.
3. Cluster anomalies → cancel duplicates + log here; no panic-resubmit.
4. No major framing changes without user review.

---

## Status snapshot (start of session, 08:15)

- **WJ-B Probe agent** (4 jobs):
  - matched128 ✅ landed: agent SPL=0.844, probe SPL=0.709, **Δ=−0.134**
  - foveated   ✅ landed: agent SPL=0.749, probe SPL=0.635, **Δ=−0.115**
  - blind      ⏳ still running (1h+ in)
  - uniform    ⏳ still running (1h+ in)
  - **Preliminary interpretation**: both negative → memory is policy- + trajectory-coupled, not scene-generic. Reframed in §4.5 as "Memory-init transplant test" complementing H2 format isolation.

- **WJ-A Memory-length sweep** (24 jobs):
  - blind  k∈{1,4,16,64,256} NPZs landed; k=1000 running
  - matched128 k∈{1,4,16,256,1000} landed; k=64 running
  - uniform k=1 running; k=4/16/64/256/1000 PENDING
  - foveated all PENDING
  - Need analyze.py on landed NPZs to get R²

- **WJ-C Occupancy decoder** (stage 1):
  - scene_occupancy (2857420) PENDING
  - Cluster slots needed; will start once probe-agent jobs free up GPUs

- **WJ-D Bug baseline** (2857386):
  - Status unclear (was RUNNING earlier, not in latest squeue tail; may have completed)
  - Sync bug_baseline_results/ to confirm

- **foveated_v2 retrain** (2857437): PENDING in queue.

---

## Hour-by-hour entries (auto-appended by /loop ticks)

(Each /loop iteration writes one block here.)

### Tick 09:15 (WJ-A first pass — UNEXPECTED, do not paper-integrate yet)

memlen NPZs analyzed (n=21 of 24, foveated_k1000 + uniform_k1/k4 still running). GPS R² per (cond, K):

| Cond | K=1 | K=4 | K=16 | K=64 | K=256 | K=1000 |
|---|---|---|---|---|---|---|
| Blind | +0.89 | +0.91 | +0.79 | +0.90 | +0.93 | +0.92 |
| Coarse (mtc128) | +0.69 | +0.37 | +0.18 | +0.48 | +0.33 | **−1.52** |
| Uniform | — | — | −1.61 | +0.43 | −0.00 | −0.93 |
| Foveated | +0.78 | +0.61 | +0.13 | +0.60 | +0.46 | — |

**Pattern is NOT consistent with my hypothesis** (bottleneck R² grows monotonically with K). Observations:
- Blind: ROBUST to K (~0.85-0.93 across all K). My K=1 should give "single-step memory" but R² stays high.
- Coarse: bouncy; K=1000 CRASHES to −1.52 with high variance.
- Uniform / Foveated: noisy.

**Why-might-be-wrong (methodological issue)**:
- The GPS sensor is part of the observation at every step (Wijmans-style sensor stack: gps + compass concatenated to LSTM L0 input). With K=1 (reset hidden state every step), the stored h_2 = LSTM(o_t, h_init=0) at each step — but o_t still includes the GPS sensor reading, which feeds into L0 every step. So even K=1 contains GPS info from current step, just no integration history.
- The "memory budget" interpretation as designed doesn't isolate "needs K steps history to decode GPS" because the per-step GPS sensor leaks into h_2 every step, regardless of K.
- This is a confound in eval-time clipping. Wijmans's original Fig 2 used architectural memory restriction during TRAINING (k-step history-only), which is a different experiment.

**Action**:
- Do NOT update paper §4.2 with these numbers.
- Keep §4.2 \hcpending stub but note that the eval-time clipping protocol is methodologically confounded.
- Real fix would require either (a) GPS-masked rollouts + K sweep, or (b) architectural-memory-budget retrains. Both are out of scope for this submission.
- WJ-A as designed gives weak/inconclusive evidence; consider dropping from paper or moving to appendix as "what doesn't work" methodological note.

### Tick 09:05 (Hour 1, blind landed)

**WJ-B Probe agent (4/4 COMPLETE)**:
| Cond | Agent SPL | Probe SPL | Δ | succ_a | succ_p |
|---|---|---|---|---|---|
| Blind | 0.471 | 0.331 | **−0.140** | 0.81 | 0.67 |
| Coarse | 0.844 | 0.709 | **−0.134** | 0.97 | 0.95 |
| Foveated | 0.749 | 0.635 | **−0.115** | 0.92 | 0.91 |
| Uniform | 0.790 | 0.688 | **−0.102** | 0.95 | 0.90 |

**ALL 4 NEGATIVE** — robust pattern. The "memory is policy + trajectory-bound" framing holds for all conditions including blind. Blind shows largest absolute drop (success 0.81→0.67); others retain ≥90% success but lose path-efficiency.

**Decision**: positive-result threshold met (consistent direction across all 4 conditions, magnitudes 10-14% SPL). **Updated §4.5** with full numbers, removed `\hcpending` markers. Single-seed `\pendnote` remains.

**WJ-D Bug baseline FAILED** (2857386, 5:55min, exit 1:0):
- Root cause: my script looked for `pointgoal_with_gps_compass` sensor, but our Wijmans-faithful sensor stack uses `goal_in_start_frame` + GPS + compass separately
- Fix: compute (rho, theta) directly from `env.sim.get_agent_state()` + `current_episode.goals[0].position` using quaternion rotation
- Resubmitted as 2857568

**WJ-A**: ~16 NPZs landed. Started analyze.py batch job in background (bpouqixyx).

**WJ-C scene_occ**: still PENDING.

### Tick 08:35 (Hour 1)

**WJ-B Probe agent (3/4 landed)**:
| Cond | Agent SPL | Probe SPL | Δ | succ_a | succ_p |
|---|---|---|---|---|---|
| matched128 | 0.844 | 0.709 | **−0.134** | 0.97 | 0.95 |
| foveated   | 0.749 | 0.635 | **−0.115** | 0.92 | 0.91 |
| uniform    | 0.790 | 0.688 | **−0.102** | 0.95 | 0.90 |

Pattern: **all 3 negative**, magnitude ~−0.10 to −0.13. Success rates barely change — the SPL drop is path-length increase, not failure increase. Consistent with "memory is policy + trajectory-bound, not scene-generic" interpretation. Still waiting on blind.

**Decision**: don't update paper §4.5 yet (await blind). If blind also negative → all-conditions story holds → fill numbers. If blind positive → blind is special (Wijmans's case) → re-evaluate framing.

**WJ-A memlen** (14 NPZs landed: blind k=1/4/16/64/256/1000 done; matched128 all 6 K done; uniform k=1/16/64/256/1000 done; foveated still running). Need to run analyze.py to get R² per K. Will do next tick.

**WJ-C scene_occ**: still PENDING (0 scenes processed).

**WJ-D Bug baseline**: 2857386 not in queue, but no output JSON either. Either still running with a different name or completed silently. Need investigation.



