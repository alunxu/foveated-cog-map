"""Parse aggregated SPL/Success metrics from a habitat-baselines eval log.

Usage: python parse_eval_metrics.py <log_path> <cond> <n_eps_target> <out_json>
"""
import json
import re
import sys

log_path, cond, n_eps_target, out_json = sys.argv[1:5]
text = open(log_path).read()

metrics = {}
# habitat-baselines aggregated stats: 'reward: X  spl: Y  success: Z'
for m in re.finditer(r"\b(reward|spl|success|distance_to_goal|num_steps|soft_spl|distance_to_goal_reward)\s*[:=]\s*([-+]?\d+\.?\d*)",
                     text):
    k = m.group(1)
    v = float(m.group(2))
    # take the LAST occurrence (final aggregate) by always overwriting
    metrics[k] = v

out = {
    "cond": cond,
    "n_episodes_target": int(n_eps_target),
    "metrics": metrics,
}
with open(out_json, "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {out_json}")
print(json.dumps(out, indent=2))
