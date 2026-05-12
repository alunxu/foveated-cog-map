# Paragraph stub for §5.4 (Implications and scope) — fill numbers when results land

The hook: the existing §5.4 already says "The cleanest falsification target is
direct: train a transformer navigator under our 5 sensor conditions; observing
the opposite pattern would falsify the principle's architecture-independence."

Our world-model probe is a *partial* architecture-agnostic test (transformer
encoder + LSTM integrator, not a transformer navigator end-to-end). Frame it
as such — preliminary, not the full falsification.

## Draft paragraph (insert before "broader implications" sentence)

> \noindent\textbf{Architecture-agnostic confirmation (preliminary).}\quad As
> a partial test of the architecture-independence claim, we ran a frozen
> DINOv2-Base patch encoder \citep{oquab2024dinov2}\uncertain{ref?} feeding a
> small LSTM trained on Memory Maze \citep{pasukonis2023memorymaze} 9$\times$9
> offline trajectories, with five resolution conditions (\texttt{blind},
> \texttt{coarse} 14$\times$14, \texttt{foveated} 56$\times$56 with
> $\sigma{=}4$ blur, \texttt{uniform} 56$\times$56, and \texttt{foveated\_logpolar})
> chosen to mirror our Habitat chassis. After 8k steps of next-feature
> prediction training, we probe \texttt{agent\_pos} from the LSTM hidden state
> with a linear and a 4-layer MLP probe (Pasukonis 2023 protocol; eval window
> steps 250--500 of $T{=}500$). The protocol, prediction shape, and decision
> rules are pre-registered (\verb|scripts/probing/world_model_probe/PRE_REGISTRATION.md|).
> Linear $R^2$ <FILL> across blind / coarse / uniform conditions (cf.\ Figure
> X), and the gap (MLP$-$linear) <FILL>; the qualitative pattern <FILL> the
> capacity-allocation prediction. We treat this as a preliminary confirmation
> on a different environment, encoder family (transformer-patch instead of
> ResNet-18), and policy regime (offline next-feature prediction instead of
> on-policy DD-PPO), rather than a full architecture-independence test --- the
> recurrent core is still LSTM. A full transformer-navigator replication
> remains the cleanest falsification target.

## Numbers slots to fill from /tmp/wmprobe_results/summary.json

* "Linear R² <FILL>": e.g. "rises monotonically from 0.05 (blind) through
  0.18 (coarse) to 0.45 (uniform)".
* "the gap (MLP − linear) <FILL>": e.g. "shrinks from 0.31 at coarse to 0.07
  at uniform".
* "the qualitative pattern <FILL> the capacity-allocation prediction": e.g.
  "matches" / "partially matches (linear monotone but gap not coarse > uniform)"
  / "does not match".

## If the result is null

Then we report it in §5.6 Limitations rather than §5.4:

> \noindent (iv)~\emph{Architecture-agnostic test was preliminary and did not
> reproduce.} A frozen DINOv2-Base + small LSTM probe on Memory Maze 9$\times$9
> at five sensor-resolution conditions did not show the predicted monotone
> bandwidth → linear-readable position-code trend. Plausible reasons: the
> environment differs (memory-maze object retrieval vs.\ Habitat point-goal
> navigation), the policy is off-policy (next-feature prediction on
> scripted-explorer trajectories) rather than on-policy RL, the encoder is
> not adapted to the task, and the LSTM is much smaller than our agents'
> recurrent core. The principle's architecture-independence remains an open
> claim; a same-architecture replication on a transformer-RL navigator is the
> cleaner test.
