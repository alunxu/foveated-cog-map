import json, os
base = "/scratch/izar/wxu/probing_results"
print(f"{'condition':<25} {'n_steps':>8} | {'gps_r2':>8} {'compass_r2':>10} {'combined':>8} | {'gps_mae_m':>10} {'comp_mae_deg':>12}")
print("-" * 100)
for name in ["blind_gibson", "uniform_gibson", "foveated_gibson", "foveated_learned_gibson", "matched_gibson"]:
    for suffix in ["", "_det"]:
        p = f"{base}/{name}{suffix}_analysis.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        g = j["1b_global_gps_compass"]
        tag = f"{name}{suffix}"
        print(f"{tag:<25} {j['n_steps']:>8} | {g['gps_r2']:+8.3f} {g['compass_r2']:+10.3f} {g['combined_r2']:+8.3f} | {g['gps_mae_m']:10.3f} {g['compass_mae_deg']:12.3f}")
    print()
