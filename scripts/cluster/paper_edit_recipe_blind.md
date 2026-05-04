# Paper edit recipe: apply blind_izar values once analysis JSONs land

After `bash scripts/cluster/run_post_blind_analyses.sh` finishes, the
following JSONs should exist on RCP `/scratch/wxu/habitat_checkpoints_rcp/analysis_results/`:

- `blind_izar_det_analysis.json` (analyze.py output: SPL/Success/GPS R²/Compass R²/eigenspectrum)
- `mlp_probe.json` (5 conds: linear+MLP+gap)
- `lagk_summary.json` (5 conds: GPS/compass/DtG × k=0,2,5,10,20)
- `skaggs_rectified.json` (5 conds: mean per-(unit, scene) Skaggs + place-unit counts)
- `cka_5cond.json`, `procrustes_5cond.json` (5×5 matrices)

Pull these JSONs locally first:
```
RUNAI_CURRENT_CTX=rcp kubectl exec -n runai-dhlab-wxu <any-Running-pod> -- \
    cat /scratch/wxu/habitat_checkpoints_rcp/analysis_results/<json> > /tmp/<json>
```

## Edits to apply (HIGH-confidence only)

### 1. Table 1 (L215, blind row)
- Frames: 340M (izar ckpt.34 trained 340M frames), not 250M
- SPL, Success, GPS R², Compass R²: from `blind_izar_det_analysis.json`
  - SPL → `1b_global_gps_compass.gps_r2_no_cap` related (need to check exact field)
  - Actually SPL/Success come from rollout summary. Check `n_episodes` and any
    `spl_mean` / `success_mean` field. If missing, may need to compute from NPZ.
- Comment out the "Blind row pending own dh-blind retrain" line in 4-cond comment block

### 2. §H1 information-allocation paragraph (L245)
Currently:
```
Linear (Ridge α=10) GPS R² varies dramatically across conditions ($+0.580$ coarse, $+0.162$ foveated, $-0.121$ foveated-logpolar, $-1.187$ uniform; range ≈ 1.8 across the four sighted conditions; blind row pending own retrain).
```
Replace blind-pending caveat with values:
```
Linear (Ridge α=10) GPS R² varies dramatically across conditions ($+X.XX$ blind, $+0.580$ coarse, $+0.162$ foveated, $-0.121$ foveated-logpolar, $-1.187$ uniform; range ≈ Y across all five conditions).
```
Same for MLP (insert blind value) and Gap (insert blind gap, expected ~0).

### 3. §Skaggs paragraph (L277)
Currently:
```
mean per-(unit, scene) Skaggs is $1.20$ (coarse), $1.24$ (foveated), $1.19$ (uniform), $1.16$ (foveated-logpolar)
```
Insert blind:
```
mean per-(unit, scene) Skaggs is $X.XX$ (blind), $1.20$ (coarse), $1.24$ (foveated), $1.19$ (uniform), $1.16$ (foveated-logpolar)
```
Plus place-unit count update.

### 4. §Lag-k SR paragraph (L286)
Currently has stale narrative "Blind shows long predictive horizon: R² ≥ 0.72 from k=0 to k=20".
After: verify against blind_izar lagk_summary.json[blind_izar][GPS][k0..k20].
If stable >0.7 across k: claim survives, update with new exact values.
If unstable: rewrite with the actual profile.

### 5. Strip L196 stale-results notice
Once Table 1 + L245 + L277 + L286 all use 5-cond values, remove:
```
\toconfirm{\textbf{Stale-results notice (internal, strip before submission).} ...}
```

## Edits to DEFER (Appendix tables; require analyses we haven't run)

- L361 Wijmans replication ECDF: needs separate eigenspectrum / variance run
- L673 Linear/MLP scrub control: needs scrub-probe run (re-train probe on shuffled features)
- L700 lag-k full Appendix table (k=0,1,5,10,20,50): we only have k=0,2,5,10,20
- L747 Eigenspectrum α exponent (Stringer): needs separate Stringer power-law fit

These can stay with stale values + explicit `\toconfirm{}` flagging; mention
in §limitations as not-yet-rerun-with-blind_izar if necessary.
