"""Summarize CV probe results for all det + stoch conditions."""
import json, os

base = "/scratch/izar/wxu/probing_results"
print(f"{'condition':<28} | {'gps_split':>10} {'gps_cv':>18} | {'comp_split':>10} {'comp_cv':>18} | {'dtg_split':>10} {'dtg_cv':>18}")
print("-" * 150)

for name in ["blind_gibson", "uniform_gibson", "foveated_gibson", "foveated_learned_gibson", "matched_gibson"]:
    for suffix in ["", "_det"]:
        p = f"{base}/{name}{suffix}_analysis.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        g = j["1b_global_gps_compass"]
        dtg = j["1c_distance_to_goal"]
        tag = f"{name}{suffix}"
        gps_split = f"{g['gps_r2']:+.2f}"
        gps_cv = f"{g.get('gps_cv_r2_mean', float('nan')):+.2f}±{g.get('gps_cv_r2_std', float('nan')):.2f}"
        comp_split = f"{g['compass_r2']:+.2f}"
        comp_cv = f"{g.get('compass_cv_r2_mean', float('nan')):+.2f}±{g.get('compass_cv_r2_std', float('nan')):.2f}"
        dtg_split = f"{dtg['r2']:+.2f}"
        dtg_cv = f"{dtg.get('cv_r2_mean', float('nan')):+.2f}±{dtg.get('cv_r2_std', float('nan')):.2f}"
        print(f"{tag:<28} | {gps_split:>10} {gps_cv:>18} | {comp_split:>10} {comp_cv:>18} | {dtg_split:>10} {dtg_cv:>18}")
    print()
