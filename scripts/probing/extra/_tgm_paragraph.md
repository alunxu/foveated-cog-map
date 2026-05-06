# Draft: TGM subsection for §3 / §4 of main.tex

## Where to insert

Best fit: as a new subsection between §3.4 (Format axis: across-condition non-
interchangeability) and §3.5 (Consumption axis), which currently jumps from
"the codes occupy different subspaces" to "probe-readability dissociates from
policy reliance" without addressing how the codes evolve in time. The TGM
panel directly bridges that gap.

## Subsection title (verdict-style, matches existing pattern)

\subsection{Format axis (temporal): the blind code rotates over time, sighted
codes stabilise}

## Draft paragraph

\noindent The format-axis evidence so far compares static snapshots --- LOSO
generalisation, Procrustes shape, subspace divergence. To ask whether the
\emph{temporal architecture} of the code differs across conditions we run a
King-Dehaene Temporal Generalisation Matrix~\citep{king2014temporal}: train a
ridge regressor of egocentric goal-vector at trajectory step \(t_{\text{train}}\)
on a held-out episode set, evaluate at step \(t_{\text{test}}\), report \(R^2\)
for every \((t_{\text{train}}, t_{\text{test}})\) pair (\(T{=}50\) steps,
\(n_{\text{eps}}\!=\!300\) per condition, PCA top-30 pre-reduction, ridge
\(\alpha{=}10\); pre-registered in \texttt{scripts/probing/extra/temporal\_generalisation.py}).
The five resulting \(50{\times}50\) matrices (Figure~\ref{fig:tgm}) reveal three
distinct temporal architectures: a clear \emph{diagonal-only} pattern in the
\textsc{blind} agent (peak \(R^2\) at \(t_{\text{train}}=t_{\text{test}}\),
off-diagonal mean \(=-0.52\); decoder accuracy at lag-30 falls to \(R^2 = -0.91\),
signalling the linear basis at step \(t\) is incompatible with the basis at
\(t{\pm}30\)); a \emph{square-block} pattern in the \textsc{coarse} 1\(\times\)1
agent (off-diagonal mean \(=+0.15\); lag-30 \(R^2 = +0.07\) --- the same linear
basis still works thirty steps later); and graded intermediate patterns in the
three sighted-rich conditions (\textsc{foveated} 4\(\times\)4 lag-30 \(=-0.08\),
\textsc{uniform} 4\(\times\)4 \(=+0.03\), \textsc{foveated-logpolar} \(=-0.15\)).
Code half-life ordered: \textsc{coarse} \(>\) \textsc{uniform} \(>\)
\textsc{foveated-logpolar} \(>\) \textsc{foveated} \(>\) \textsc{blind}.

\noindent The mechanism this picture is consistent with is that \textsc{blind}
must \emph{integrate} motor signals from \(t{=}0\) to compute its goal-vector,
so the meaning of every \(h_t\) coordinate shifts step-by-step (a recurrent
basis rotation); \textsc{coarse}, by contrast, derives goal-direction from a
slowly-varying colour cue at every step, so the \(h{\to}\)goal-vec mapping is
quasi-stationary. The other three sighted conditions interpolate. The TGM thus
offers the consumption-axis dissociation our static probes do not: \textsc{coarse}
agents \emph{read} the same goal direction across most of the trajectory;
\textsc{blind} agents \emph{recompute} it from the action sequence
each step.

\uncertain{The pre-registered prediction was the opposite (\textsc{blind} as
working-memory \emph{block}, \textsc{coarse} as recompute-each-step diagonal);
we report what we found.}

## Note for §5 (Discussion)

The TGM result tightens the bio-AI bridge to King-Dehaene MEG decoding (the
canonical tool for distinguishing transient from sustained cortical codes in
human EEG/MEG). Pair with the Pasukonis 2023 reference and the existing Sanders
2020 / Dorrell 2024 cites.

## Caveats to disclose

- Goal-vec target is in agent (egocentric) frame; allocentric world-position
  TGM did not produce significant signal (decoder fails across scenes; saved
  to docs/manuscript/fig/cogneuro/tgm_pos_results.npz).
- T=50 covers ~50% of average episode length for sighted; covers full first
  half of typical blind episodes (median len ~120-200).
- Single-seed result; multi-seed grid would sharpen confidence intervals.

## Figure caption

\begin{figure}
  \centering
  \includegraphics[width=\linewidth]{fig/cogneuro/fig_tgm.pdf}
  \caption{\textbf{Temporal Generalisation Matrix shows blind agent uses a
  transient code, coarse agent a sustained code.} Each panel is a
  \(50{\times}50\) matrix: the \((i,j)\) cell is the \(R^2\) of an
  egocentric-goal-vector ridge decoder fit at step \(i\) and evaluated at
  step \(j\), 5-fold cross-validated across 300 episodes, PCA top-30
  pre-reduction. Diagonal-only structure (\textsc{blind}) means the linear
  read-out for goal direction rotates over time; square-block structure
  (\textsc{coarse}) means the same linear projection works throughout the
  trajectory. Sighted-rich conditions interpolate. Method: King \&
  Dehaene 2014~\citep{king2014temporal}; protocol pre-registered in
  \texttt{scripts/probing/extra/temporal\_generalisation.py}.}
  \label{fig:tgm}
\end{figure}
