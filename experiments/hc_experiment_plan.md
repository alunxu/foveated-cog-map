# HC cluster experiment plan — friend's 14 trainings (with rationale)

**For**: friend running on hc cluster (1× H200 + 4× A100).
**Deadline**: 2026-05-06 (NeurIPS submission).  **Today**: 2026-04-27.
**Net working days**: ~9.

**Hardware note**: A100 is ~1.5–2× slower than H100 on this workload, so
the per-training wall times below (originally written for H100) may
extend.  Realistic budget: H200 trainings 2–3 days, A100 trainings
**3–4.5 days**.  Re-prioritise downward: if Tier 3 doesn't fit, drop
it.  Tier 1 must finish; Tier 2 strongly preferred.

This document is the single source of truth for what to run, why, and how to
ship results back.  It is verbose on purpose: when a training fails or the
prediction comes out unexpected, the *why-paragraph* tells you whether to
panic, retry, or just log it.

---

## TL;DR — what's at stake

The paper claims a **substitution mechanism (H1)**: when the visual encoder
produces enough information per step, the LSTM stops integrating GPS, and
the linear top-layer GPS code disappears.  Rich-encoder agents (uniform,
foveated) lose linear GPS at $\mathbf{h}_2$; bottleneck agents (blind,
coarse) preserve it.  So far this rests on **single-seed** results across
**5 fixed conditions**.  Reviewers will ask:

1. **Is this seed-1 noise?** → multi-seed replication (3 conditions).
2. **Is encoder bandwidth really the axis?** → scaling sweep at
   $K \in \{32, 64, 96, 128, 192\}$ input resolution.
3. **What if foveation isn't blur but spatial subsampling?** → log-polar
   falsifiable test.
4. **What about dynamic gaze (H3)?** → stochastic gaze policy.
5. **What about the foveated condition's NaN-corrupted ckpt.36?** →
   foveated_v2 clean re-run.

The 14 trainings below answer those five questions.  Tier 1 (5 trainings)
covers the must-haves; Tier 2 (5) fills out the scaling sweep + falsifiable
test; Tier 3 (4) is foveation completeness if time allows.

ETA per training: **2–3 days on H200**, **3–4.5 days on A100** (vs ~5 on V100).

---

## 处理 unexpected results 的原则（中文）

每个 training 下面都有 explicit 的预期数值（"Specific prediction"）。结果可能落在 4 类：

1. **完全符合预期** → 整合进 paper（已有 stub 位置）。
2. **方向对、量级偏差**（e.g. R² 预期 0.5–0.7 实际 0.3） → 多 seed / 多 ckpt 验证后用 mean ± std 报告。
3. **方向反 / falsified** → ⚠️ **不要直接写 paper**：先 investigate 原因，弄清 mechanism 后才决定怎么写。
4. **NaN / collapse / 训练失败** → 先排除 cluster artifact（改 seed / 改 hardware 重跑），再往 mechanism 方向挖。

**第 3 类是最容易出错的**，给出 4 条铁则：

- ❌ 不要直接把 negative 写成 paper limitation paragraph → 让 reviewer 觉得我们 hedging。
- ❌ 不要改 paper claim 来 fit 单点新结果 → over-fitting to single data point。
- ❌ 不要 over-hedge 主 claim 弱化整篇 paper → 主 claim 是基于 7+ converging 实验的，单个 anomaly 不应该撼动它。
- ✅ 跑 additional experiments to understand the mechanism → 等 mechanism 清楚再决定 paper scope（限定 / 重写 / 维持原 claim + caveat）。

**判断 anomaly 是 single-point noise 还是真 mechanism 错的标准**：
1. 用第二个 seed 重复 — 若一致 → 真信号
2. 跑 control / sanity check — 排除实现错误
3. 跑邻近的 sweep 点 — 若是 isolated outlier vs 系统偏差，画出来明显

每个 training 下面的 **「中文 — 不符合预期 investigation 协议」** block 给出该 training 的具体调查步骤。看到 unexpected 时先按那条协议跑实验，**别动 paper**。

---

## What's already done — do NOT relaunch

**On Izar (V100, our side)** — these are converged or in progress; friend
should not duplicate them.

| Status | Run | Frames | What it gives the paper |
|---|---|---|---|
| ✅ Converged | `blind_gibson` (seed=1) | 342M | Fig 2 H1 baseline, R²(GPS\|h₂)=0.95 |
| ✅ Converged | `matched_gibson` (seed=1, "coarse" 48×48 → 1×1) | 250M | Fig 2 H1 anchor, R²=0.78 |
| ✅ Converged | `uniform_gibson` (seed=1) | 250M | Fig 2 H1 anchor, R²≈0 |
| ✅ Converged | `foveated_gibson` (fix, seed=1) | 174M | Fig 2 H1 anchor, R²≈0 |
| ✅ Converged | `foveated_learned_gibson` (seed=1) | 250M | Fig 2 H1 anchor, learned-gaze |
| ✅ Converged | `matched128_gibson` (seed=1) | 250M | Scaling sweep K=128 anchor (already probed) |
| 🔄 Running | `uniform_gibson_seed=2` | 140M / 250M | Multi-seed for uniform (~6 days remaining) |
| 🔄 Running | `foveated_gibson_seed=2` | 130M / 250M | Multi-seed for foveated (fix) (~6-9 days) |

> ⚠️ **K=128 is done**.  Tier 1 covers K=32 and K=64; Tier 2 covers K=96 and
> K=192.  K=48 is the "coarse" condition (already done as `matched_gibson`).
> So the full sweep is K ∈ {32, **48**=coarse, 64, 96, **128**, 192, **256**=uniform}
> with **bold = on Izar**, the rest = friend.

---

## Tier 1 — Day 1 (5 trainings, must-have)

### Training 1: Stochastic gaze (`foveated_stochastic_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_stochastic_gibson` |
| GPU | H200 if available (most novel architecture); else A100 |
| Wall | ~2–3 days on H200, ~3–4 days on A100 @ 250M frames |
| Submit | `sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_foveated_stochastic_gibson` |

**Motivation**:  Our H3 hypothesis is that *gaze dynamics* — not just static
gaze location — should affect whether substitution occurs.  Currently the
paper has only the *static* foveated and foveated-shifted (gaze locked at
the image center or at (0.49, 0.62)).  We need a *dynamic* gaze policy whose
gaze actually moves at rollout time.

The deterministic learned-gaze MLP we tried earlier collapses to a
near-constant gaze under PPO (Appendix on H3 documents two failed pilots).
The fix is a stochastic policy: gaze is sampled from a bounded Gaussian
$\mathcal{N}(\mu, \sigma)$ with $\sigma \in [0.05, 0.30]$ per environment,
giving it a permanent exploration floor.  See
`src/habitat/foveated_stochastic_policy.py`.

**Specific success criterion (write before training finishes!)**:

* **Per-env $\sigma > 0.05$ at convergence** → policy actually keeps gaze
  diverse → H3 is testable.
* **SPL ≥ 0.7 by 250M frames** → the gaze noise didn't ruin the navigation.
* If both true, then the §4.6 H3 question becomes a real comparison:
  static-foveated vs.\ stochastic-foveated R²(GPS\|h₂) and SPL drops.

**What it would mean**:

* Stochastic R² high (~0.5+) like coarse → gaze *dynamics* prevent
  substitution → "the encoder is unreliable per-step, so LSTM keeps GPS".
* Stochastic R² low (~0) like uniform → gaze dynamics *don't* save GPS →
  substitution is about average encoder bandwidth, not per-step variability.

Either outcome is publishable.  This is the highest-information experiment
on the list.

**Watch for**: if $\sigma$ collapses to 0.05 (its lower bound) and stays
there, the policy has effectively reverted to deterministic gaze — flag to
wxu before continuing.  This was a known failure mode in earlier pilots.

#### 中文 — 不符合预期 investigation 协议

预期：per-env σ > 0.05 at convergence + SPL ≥ 0.7。

若 **σ pinned 到 0.05**（lower bound，policy 退化为 deterministic）：
- 先 sanity check entropy bonus weight 是否 zero、σ output 的 gradient path 是否被 detach（`src/habitat/foveated_stochastic_policy.py`）
- 跑 σ_max ∈ {0.10, 0.20, 0.30} 三组 ablation 找有 surviving 多样性的 regime
- ❌ 不要写 "stochastic gaze 不能学" 当 paper finding —— 我们只能说 reparam-Gaussian 这一具体路径不行
- ✅ 若三组都 collapse → §App H3 写 method-level limitation（"在 reparam-Gaussian 实现下 H3 testable 不了"），主 H3 narrative **不动**

若 **SPL < 0.5**（noise 太大干扰导航）：
- 跑 σ_max=0.05 极小噪声 ablation 看是否恢复 SPL → 用来区分 "noise-too-high" vs "policy-broken"
- 若 noise-too-high → retrain with reduced σ_max upper bound
- 若 policy-broken（SPL 仍低）→ 与 deterministic foveated baseline 对比，定位是 architecture 还是 value-function 出问题

若 **R²(GPS\|h₂) ≈ 0**（H3 实质 negative）：
- 这是 paper 最敏感的结果之一，**多跑一组 confirm**：seed=2 + σ_max=0.20 第二个配置
- 两次都 negative **才** 在 §4.6 carefully 写："reparam-Gaussian 这条路径下 H3 negative；不排除 attention-based gaze 等其他形式仍能产生效果"
- ❌ 不要把它扩成 "动态 gaze 普遍不能阻止 substitution" 的强 claim → 单 architecture 的 negative 不够推

---

### Training 2: Scaling sweep K=64 (`matched64_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_matched64_gibson` |
| GPU | A100 |
| Wall | ~3–4 days @ 250M frames on A100 |
| Submit | `sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_matched64_gibson` |

**Motivation**:  The H1 substitution mechanism predicts that **encoder
spatial-output dimensionality** is the trigger axis.  A $64{\times}64$ RGB
input fed through ResNet-18 produces a $\sim 2{\times}2$ feature map.  This
sits between coarse's $1{\times}1$ (full GPS preservation) and uniform's
$8{\times}8$ (full substitution).

**Specific prediction (paper §App E)**:  R²(GPS\|h₂) somewhere in
$[\sim 0.4, 0.6]$ — partial preservation.  SPL between coarse's 0.84 and
uniform's 0.79 (likely ~0.82).

**What it tests**:  Whether the H1 axis is *smooth* (R² declines
monotonically as K rises) or *threshold-like* (R² stays high until some
critical K then drops).  This is the headline finding of the scaling sweep
appendix and the falsifiability core of H1.

**What it would mean if R² jumps directly from 0.78 (K=48) → ~0 (K=64)**:
the substitution mechanism is closer to a phase transition than a
gradient — paper framing would shift.

**What it would mean if R² stays near 0.78**: the relevant axis isn't
encoder spatial output but something else (input pixel count? channel
information?) — would force a re-think.

#### 中文 — 不符合预期 investigation 协议

预期：R² ∈ [0.4, 0.6]，介于 K=48 (0.78) 和 uniform (≈0)。

若 **R² ≈ 0**（与 uniform 一样，跨过中间）：
- 先 probe encoder feature map 实际维度（应该是 2×2，但要测）
- 若实测 1×1（encoder 提前 collapse）→ K=48..64 之间 encoder bandwidth 都是 1×1，那两点 R² 应该都接近，与观察不符 → 重新 frame axis（不是 input res，是 encoder output dim）
- 若实测 2×2 但 R² 仍 ≈ 0 → mechanism 是 phase-transition 而非 smooth；进一步跑 K=72 / K=80 在拐点附近 isolate 转变
- ❌ 不要画 "K=48: 0.78, K=64: 0.0" 当主 figure，让 axis 看起来 cliff
- ✅ 等 K=72 + K=80 拼出过渡区再决定 figure

若 **R² ≈ 0.78**（与 coarse 一样，无下降）：
- 说明 K=64 仍在 bottleneck regime → 跑 K=80, 96, 112 找 R² 开始降的临界点
- ❌ 不要在 paper 里画 K=48 + K=64 同高度的 bar 而忽略 K=80+
- ✅ 等 K=96 / K=128 落地拼出完整曲线再 finalize figure

若 **non-monotonic**（K=64 R² > K=48）：
- seed=1 outlier 嫌疑大 → 跑 seed=2 K=64 confirm
- ❌ 单点 anomaly 不要 paper integrate
- ✅ seed=2 一致才报；不一致就 appendix flag "K=64 high seed variance, point dropped from main curve"

---

### Training 3: Scaling sweep K=32 (`matched32_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_matched32_gibson` |
| GPU | A100 |
| Wall | ~3–4 days @ 250M frames on A100 |
| Submit | `sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_matched32_gibson` |

**Motivation**:  K=32 produces a $1{\times}1$ encoder feature map (same as
coarse's K=48), but with *even lower input resolution*.  This is the
**bandwidth lower bound** of the encoder spatial-output axis.  K=32 + K=48
(coarse) jointly check whether the $1{\times}1$ collapse is the active
ingredient or whether further input-pixel reduction matters.

**Specific prediction**:  R²(GPS\|h₂) ≥ coarse's 0.78 (because the encoder
is at least as bandwidth-starved as coarse).  SPL likely a bit lower than
coarse's 0.84 (the agent has fewer pixels to even check for collisions).

**What it would mean**:

* R² ≈ 0.78–0.95 → substitution depends on encoder spatial-output, not on
  input resolution per se → confirms the mechanism's "rate axis" framing.
* R² much lower than coarse's 0.78 → low input resolution by itself is enough
  to reduce GPS retention even when the encoder bottleneck is the same →
  weakens the "encoder-output is the trigger" claim, would force re-framing.

This is the **lower-end anchor** for the scaling sweep figure.

#### 中文 — 不符合预期 investigation 协议

预期：R² ≥ 0.78（甚至更高，因为 input 比 K=48 更 starved）。

若 **R² 显著 < 0.78**（preserve 弱于 coarse）：
- 先看 SPL trajectory：若 SPL < 0.7 → 没收敛，多训 50M 帧再 probe（K=32 输入小，ResNet-18 接受度可能更慢）
- 若 SPL ≥ 0.8 但 R² 仍 < 0.5 → 真有 mechanism 含义：input resolution 本身能影响 LSTM GPS 编码，**不只是** encoder bandwidth
- 若 (b) 成立，跑 K=24（如有 config）或在 paper 里 limit claim："encoder spatial output 在 K ≥ 48 是主导 axis；K < 48 时 input resolution 加入"
- ❌ 不要直接 conclude "scaling axis is wrong" → 单条 K=32 数据撼动不了 K=48 + 4 个高 K 已观察到的 trend
- ✅ paper 暂留 K=32 作为 open data point，appendix 报告，主 figure 不画

若 **训练 diverge / NaN**：
- 32×32 输入对 ResNet-18 太小，某层 spatial dim 可能已经 ≤ 1
- 检查 numeric stability；必要时 patch encoder 处理 1×1 intermediate
- 这是 implementation 问题，非 mechanism 问题，修了重跑就行

---

### Training 4: Multi-seed blind seed=2 (`blind_gibson seed=2`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_blind_gibson` (with `seed=2` override) |
| GPU | A100 |
| Wall | ~4–5 days @ 342M frames on A100 (blind takes more frames; consider H200 if available) |
| Submit | `sbatch <flags> scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_blind_gibson 2` |

**Motivation**:  Every cross-condition R² number in the paper is **single
seed**.  Reviewers will pounce on this — Wijmans 2023 reported R²=0.95 for
blind at seed=1, we replicate seed=1=0.95, but a full peer-reviewed claim
needs at least **mean ± std over 2 seeds** so reviewers know the number isn't
a 1-in-5 lucky run.  Blind is the cleanest baseline (no visual confound), so
it's the easiest seed=2 sanity check.

**Specific prediction**:  R²(GPS\|h₂) at seed=2 within [0.85, 0.95].
SPL at seed=2 within [0.45, 0.55].  If both fall in those ranges → seed
robustness confirmed → main paper §4.1 number gets `±std` upgrade.

**What it would mean**:

* R² in [0.85, 0.95] → seed-1 wasn't a lucky outlier → cleanly publishable.
* R² < 0.7 → seed-1 was unrepresentative → trigger panic, possibly run seed=3.
* R² inside expected range but SPL much lower → blind seed-2 didn't fully
  converge — extend training before probing.

#### 中文 — 不符合预期 investigation 协议

预期：R² ∈ [0.85, 0.95]，与 seed=1 (0.95) 相差 ≤ 0.10。

若 **R² < 0.7**（与 seed=1 显著不一致）：
- 先确认是否真的 converge：blind 需要 ~342M 帧到 SPL=0.47，less-frame 会让 R² 偏低
- 看 TB success curve，若 plateau 才 probe；plateau 后仍 < 0.7 → 跑 seed=3
- 三个 seed mean ± std 报告：
  - 若 std ≤ 0.10 → "blind R² = 0.85 ± 0.05"，主 claim 不动
  - 若 std > 0.20 → carefully 写 §4.1：blind 的 single-seed R² 高度变量，**主 narrative 改用 "blind preserves a strong linear GPS code (multi-seed range 0.6–0.95)" 而非具体数值**
- ❌ 不要 paper 里直接写 "seed-1: 0.95, seed-2: 0.6"（让 reviewer 觉得 seed-1 是 outlier）
- ❌ 不要把 outlier seed 算进 mean 然后报"mean ≈ 0.78"假装一致
- ✅ honest 报 per-seed range + mean ± std

若 **SPL < 0.40**：
- 没收敛，extend training 100M 帧；若 walltime 不够，退而求其次报 ckpt.49 已有的，但 paper 里 caveat 写明帧数

---

### Training 5: Multi-seed matched seed=2 (`matched_gibson seed=2`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_matched_gibson` (with `seed=2` override) |
| GPU | A100 |
| Wall | ~3–4 days @ 250M frames on A100 |
| Submit | `sbatch <flags> scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_matched_gibson 2` |

**Motivation**:  Same as blind seed=2, but for the *coarse* (matched 48×48
→ 1×1) condition.  Coarse is the most theoretically interesting baseline:
it has visual input but no usable encoder spatial output, so R²(GPS\|h₂) is
high *despite* having vision.  If coarse's R²=0.78 doesn't replicate at
seed=2, the entire "encoder spatial-output is the trigger" mechanism story
weakens.

**Specific prediction**:  R²(GPS\|h₂) at seed=2 within [0.65, 0.85].
SPL at seed=2 within [0.80, 0.90].

Together with blind seed=2, this gives us multi-seed numbers for **both
ends of the bottleneck regime**.  With foveated_learned seed=2 added in
Tier 2, we'll have 3-of-5 conditions with seed-2 — sufficient for an "all
H1 numbers replicate to within ±0.10 across seeds" caveat upgrade.

#### 中文 — 不符合预期 investigation 协议

预期：R² ∈ [0.65, 0.85]，SPL ∈ [0.80, 0.90]。

若 **R² < 0.5**（远低于 seed=1 的 0.78，比 blind seed=2 anomaly 更严重）：
- coarse 比 blind 收敛快，250M 应足够，**所以 R² 偏低不太可能是 undertraining**
- 跑 seed=3 confirm；若仍 < 0.5 → coarse 的 R² 高度 seed-dependent
- 这是对 mechanism 的 serious 挑战：paper 主 claim 是 "coarse retains GPS like blind"
- ❌ 不要把 seed=1 + seed=2 算 mean 掩盖 seed=2 异常
- ❌ 不要直接弱化 H1 主 claim → coarse 即使 seed-noisy 仍 ≥ 0.5，与 uniform 的 ≈0 仍有 1σ 以上 separation
- ✅ 报 per-seed + 三 seed std；若 std > 0.20 → §4.1 改写："coarse 显示 LSTM GPS code 持久存在 (seed-mean R²=0.5–0.8)，但 seed variance 比 blind 大，说明在 coarse 这种边际 bottleneck regime 训练动力学比 blind 更敏感"

若 **R² ≈ 1.0**（异常高，超出预期上界）：
- 可能 episode 不够 diverse → probe overfit
- 跑 5-fold CV 看 σ；若 σ 极低且各 fold 都 ≈ 1.0 → 真信号但需 sanity check probe 没 leak
- 若 σ 大 → probe instability，不能信单 fold 的高分

---

## Tier 2 — Day 3-4 (5 trainings, when Tier 1 frees up GPUs)

### Training 6: Scaling sweep K=96 (`matched96_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_matched96_gibson` |
| GPU | A100 |
| Wall | ~3–4 days on A100 |

**Motivation**:  Mid-sweep filling.  K=96 produces $\sim 3{\times}3$
encoder output.  Together with K=64 (~$2{\times}2$) and K=128 (~$4{\times}4$)
this gives a 3-point trace through the regime where R² should be smoothly
declining if the H1 axis is gradient-like.

**Specific prediction**:  R²(GPS\|h₂) in $[\sim 0.2, 0.4]$.

**What it tests**:  smoothness of the K → R² curve.  If K=64 → 0.6, K=96 →
0.3, K=128 → 0.1 — the curve is clean.  If K=64 → 0.6, K=96 → 0.5, K=128 →
0.1 — there's a knee, suggests a phase transition, paper figure caption
adjusts accordingly.

#### 中文 — 不符合预期 investigation 协议

预期：R² ∈ [0.2, 0.4]，介于 K=64 和 K=128 之间。

若 **R² > 0.5**（mid-sweep 反弹）：
- 检查 K=96 训练是否真的收敛到 SPL ≥ 0.85（与 K=64 / K=128 一致）
- 若 SPL 异常低 → undertraining，多训 50M 帧
- 若 SPL 正常但 R² 高 → mechanism 不严格 monotonic，可能在 K=96-128 之间有 knee
- ❌ 不要 paper 里隐藏这个点假装 monotonic
- ✅ 把 K=96 当 informative outlier，appendix 单独讨论；主 figure 用 mean ± std 让 reader 自己判断 axis 平滑性

若 **R² ≈ 0**（比预期更早降到 floor）：
- monotonic 但下降比预期快 → 在 K=80 附近有相变
- 跑 K=80 拉开拐点，paper figure 在 K=64-96 之间画过渡区注解

---

### Training 7: Scaling sweep K=192 (`matched192_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_matched192_gibson` |
| GPU | A100 |
| Wall | ~3–4 days on A100 |

**Motivation**:  High-end of the scaling sweep (close to uniform's K=256).
K=192 → ~$6{\times}6$ encoder output.  Tests whether R² has *bottomed out*
near uniform's $\approx 0$ by K=192, or whether it's still falling into
uniform.

**Specific prediction**:  R²(GPS\|h₂) in $[0, 0.15]$ — basically at-floor.

**What it tests**:  upper anchor of the scaling sweep figure.  Confirms
that the **R² → 0 regime** isn't just a uniform-specific quirk (different
visual style) but a continuous extrapolation of the matched-K series.

#### 中文 — 不符合预期 investigation 协议

预期：R² ∈ [0, 0.15]，接近 uniform floor。

若 **R² > 0.3**（仍残留 GPS code，未到 floor）：
- 怀疑 substitution 在 K=192 还没完成 → 多训 50M 帧再 probe，看 R² 是否下降
- 或者 6×6 encoder output 的 substitution 比 uniform 的 8×8 慢 → 与 uniform 中间还有 K=224 这个空间
- ❌ 不要直接 paper integrate（破坏 sweep 平滑性）
- ✅ 跑 K=224 中间点确认是否 K=192 是 isolated outlier 还是真的需要 K ≥ 224 才达 floor；若是后者，paper 写"substitution completes around K ≈ 200 in our setup"

若 **R² ≈ 0**（符合预期）但 σ 大：
- σ > 0.5 是 rich-encoder 区典型 → 用 mean ± std 报告，不用单点

---

### Training 8: Multi-seed foveated_learned seed=2 (`foveated_learned_gibson seed=2`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_learned_gibson` (with `seed=2`) |
| GPU | A100 |
| Wall | ~3–4.5 days @ 250M frames on A100 (or 2–3 if scheduled on H200) |
| Submit | `sbatch <flags> scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_foveated_learned_gibson 2` |

**Motivation**:  Foveated_learned is the **H3 anchor** in the paper — gaze
is predicted by an MLP head trained end-to-end.  Seed=1 gave R²=0.67 ±0.18
(early-train), declining in a manner specific to learned-gaze.  Multi-seed
needed to confirm the *decay-rate ordering* (uniform fastest, foveated
slower, foveated_learned in between) — currently the part of the paper
"most exposed to seed variability" by our own admission.

**Specific prediction**:  R²(GPS\|h₂) at seed=2 within [0.55, 0.75]
early-training, declining to chance later.  Decay timing should be similar
to seed=1 (peak around 50M frames, decay by 100-150M).

**What it would mean**:

* Seed=2 decay roughly matches seed=1 → "rich-encoder substitution
  timescale" claim survives multi-seed → keep §4.4 narrative.
* Seed=2 decay much faster or slower → substitution timescale is
  noisy across seeds → soften the "uniform fastest, foveated_learned
  middle" claim, possibly drop the per-condition decay-rate ordering.

This is the multi-seed condition with the **biggest paper-impact risk** —
explicitly flag results to wxu.

#### 中文 — 不符合预期 investigation 协议

预期：early-train R² ∈ [0.55, 0.75]，~150M 帧后 decay 到 floor。

若 **decay 速度与 seed=1 显著不同**（>50% 时序差异）：
- 这是 paper 最敏感的 multi-seed claim — §4.4 "decay-rate ordering uniform > foveated_learned > foveated"
- 立刻收集 per-ckpt probe data（10/20/30/40/49 全跑）画 dual-seed 时间曲线
- 若 ordering 在两 seed 间不稳定 → 主文 §4.4 关于 decay-rate 的 fine-grained sentence **删除**（这是 within-condition seed noise，不是 condition-level pattern）
- 主文留 "encoder-richer agents lose GPS faster" 的 binary claim（uniform/foveated_learned vs blind/coarse），**删除** 三 rich-encoder condition 间相对快慢的 ordering
- ❌ 不要给 ordering 加一个 "single-seed" footnote 然后保留 → reviewer 会 pinpoint
- ✅ honest 删除 fine-grained ordering，保留 coarse-grained binary claim

若 **early-train R² < 0.4**（弱于 seed=1 的 0.67）：
- 检查 learned-gaze 是否收敛到合理位置 → 看 gaze trajectory
- 若 gaze 退回 (0.5, 0.5) → 与 foveated_fix 等价，结果合理（两个 seed 都收敛到 center 在情理之中）
- 若 gaze 收敛到不同位置但 R² 低 → 异常，跑 seed=3 + 收集 per-ckpt 看 substitution 时序

---

### Training 9: Foveated log-polar (`foveated_logpolar_gibson`) — F3 falsifiable

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_logpolar_gibson` |
| GPU | A100 |
| Wall | ~3–4 days on A100 |

**Motivation**:  This is the **falsifiable core of H1**.  Gaussian-blur
foveation (our standard `foveated_gibson`) preserves the encoder's
$8{\times}8$ spatial output even though peripheral content is blurred.
Log-polar foveation is *spatial subsampling* — non-uniform retinal grid
that drops the encoder spatial output to $\sim 2{\times}2$.  Per the H1
mechanism (encoder spatial-output → memory recruitment), log-polar should
behave *between coarse and uniform*, not like uniform.

**Specific prediction (written before result available!)**:
$$
R^2(GPS|h_2) \in [0.30, 0.65]
$$

**Falsifiable outcome**:

* R² ≥ 0.30 → mechanism survives → §App E remains the falsifiable
  prediction it's framed as.
* R² < 0.30 (matches uniform) → **mechanism is wrong** → encoder spatial
  output is *not* the trigger → would force a paper rewrite.

Wxu wrote the prediction down before training started so reviewers can see
this is a real falsifiable test, not post-hoc.

#### 中文 — 不符合预期 investigation 协议

预期：R² ∈ [0.30, 0.65]，**falsifiable lower bound: 0.30**。

若 **R² < 0.30**（mechanism falsified — 与 uniform 一样）：
- ⚠️ 这是整篇 paper 的 falsification，**先**别 paper-rewrite，**先** 三步 sanity check：
  1. **Implementation check**: 跑 `tests/test_torch_foveation.py` + 视觉检查 log-polar 输出图像（应有显著的 ρ-θ 网格 artifact，与 Gaussian blur 完全不一样）
  2. **Encoder dim check**: 实测 encoder feature map shape — 预期 2×2，但若是 8×8（grid size 配错）→ 实际并未实现 spatial subsampling，直接重跑 with 正确 grid (n_rho=64, n_theta=64)
  3. **Seed check**: 若 (1)+(2) 都对，跑 seed=2 confirm
- 若三步都 confirm R² < 0.30 → **真 falsified**，按以下方式改 paper（不是 panic 重写）：
  - §App E falsifiable section 留下原预测 + 实测 + 承认 falsified（这个 honest reporting 实际上**加分**而非减分 —— "我们提前写下预测后被推翻，说明 mechanism 是 falsifiable 的，需要更细致的 axis 描述"）
  - §H1 主 mechanism 段落改写：把 "encoder spatial output dimensionality" 软化为 "encoder feature variety per step"（保留 mechanism 直觉但 admit 准确 axis 待定）
  - propose 后续实验（直接 measure encoder feature variety vs spatial dim）
- ❌ 不要 silently drop log-polar 段落假装没做过
- ❌ 不要保留 "encoder spatial output is the trigger" claim 仅加 footnote 否认
- ❌ 不要因为这一个 negative 就把 H1 主 claim 完全 rewrite —— H1 已被 7+ 收敛实验支持

若 **R² > 0.65**（超出 coarse 的 0.78）：
- 与 coarse 一致甚至更强 → mechanism 强版本被 confirm，integrate 顺利
- 仍要 sanity check encoder feature map 实际 shape

---

### Training 10: Foveated v2 clean re-run (`foveated_v2_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_v2_gibson` |
| GPU | A100 |
| Wall | ~3–4.5 days @ 250M frames on A100 (or 2–3 if scheduled on H200) |

**Motivation**:  Our seed=1 `foveated_gibson` ckpt.36 (174M frames) was
hit by a silent NaN-gradient corruption mode in DD-PPO that we discovered
late in the project (Appendix on training stability).  We patched the bug
and need a *clean* foveated re-run to verify that the H1 numbers
(R²(GPS\|h₂)≈0, SPL=0.75, MP3D shift) don't shift meaningfully in a
NaN-free environment.  This addresses §5.5(ii) limitations.

**Specific prediction**:  R²(GPS\|h₂) within [-0.1, +0.1] (still at floor).
SPL within [0.72, 0.80].

**What it would mean**:

* Numbers within ±0.05 of ckpt.36 → NaN bug didn't materially affect H1
  conclusions → safe to keep our results.
* Numbers shift by >0.1 → NaN bug was load-bearing → re-run on Izar with
  the clean ckpt becomes the canonical foveated number, paper figures get
  updated.

#### 中文 — 不符合预期 investigation 协议

预期：R² 与 ckpt.36（NaN-bug 前）相差 ≤ 0.05；SPL 相差 ≤ 0.05。

若 **R² 偏移 > 0.10**（尤其是从 ≈0 跳到 >0.3）：
- NaN bug 是 load-bearing → 之前 paper 上 foveated 那条线**不可靠**
- **必须**做 per-ckpt 序列对比：v2 的 ckpt.10/20/30/40/49 vs 原 buggy 的 ckpt.10/20/30/36，找 divergence point
- 替换 paper 主表 + Fig 3 substitution dynamics 中的 foveated 线为 v2 数据
- ❌ 不要 silently 用新数 不提 ckpt.36 → reproducibility 灾难，reviewer 会问"为什么这次不一样"
- ✅ §5.5(ii) limitations + appendix dedicated section 解释 NaN-bug + v2 fix 的完整 timeline；明示 v2 数据替换 ckpt.36 数据，Fig 3 标注 "(from v2, NaN-fixed)"

若 **SPL 偏移 > 0.05**（训练 stability 改变了）：
- 怀疑 A100 vs V100 hardware artifact
- 跑 sanity check：用同一个 ckpt 在 A100 和 V100 上做 forward pass，看 logits 是否 bit-exact
- 若 hardware 一致但 SPL 仍偏 → 就是 NaN-fix 影响了 training dynamics（合理的，bug 修了行为变了）
- paper 更新所有 foveated SPL number；尤其 §4.5 transplant、shortcut analysis 都要重做

若 **R² 与 ckpt.36 一致**（符合预期）：
- 写一句 "v2 confirms ckpt.36 numbers within ±0.05" + appendix 表格对比；ckpt.36 仍作 paper 主 ckpt 不动

---

## Tier 3 — Day 6-7 (4 trainings, foveation completeness; skip if running tight)

### Training 11: Foveated σ=20 (`foveated_strong_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_strong_gibson` |
| GPU | A100 |
| Wall | ~3–4 days on A100 |

**Motivation**:  F4 *foveation strength* sweep, **high-blur endpoint**.
At σ_max = 20 the periphery is so blurred that the encoder's $8{\times}8$
output is effectively a center-only spotlight.  Tests whether stronger
blur produces uniform-like substitution (encoder dominates) or coarse-like
preservation (encoder so weak it can't substitute).

**Specific prediction**:  R²(GPS\|h₂) in $[0.3, 0.6]$ — partial preservation,
between coarse and standard foveated.

**Why it matters**:  Foveation under our model is currently single-
strength (σ=8).  A 4-point strength sweep (σ ∈ {2, 4, 8, 12, 20} with σ=4
explicitly skipped, see below) gives a continuous knob into the H1
substitution dynamics within the foveation family.

#### 中文 — 不符合预期 investigation 协议

预期：R² ∈ [0.3, 0.6]（mid-bottleneck）。

若 **R² ≈ 0**（与 standard foveated 一样，σ=20 没能 push 到 bottleneck）：
- 视觉检查 σ=20 输出（peripheral 是否真的高度模糊到看不出结构）
- 若视觉看上去 OK，说明 Gaussian blur 这个 model class 即使到 σ=20 也保留太多 spatial info → **支持** "log-polar 必要" 的论述（与 T9 形成对照）
- ❌ 不要 paper 里写 "stronger blur didn't help" 当孤立主 claim → 与 T9 log-polar 一起讨论才有解释力
- ✅ §App E 写 σ-sweep flat 在 R²≈0 + log-polar at R²≈0.5 → 两实验 jointly 说明 "blur strength alone insufficient; spatial subsampling necessary"

若 **R² > 0.7**（接近 coarse）：
- σ=20 把 encoder 推到 1×1-like regime → 强支持 mechanism
- 但要 sanity check：encoder feature map 实际维度是不是真的退化到 ~2×2
- 与 K=64 的 R² ≈ 0.5（也是 ~2×2 encoder）对比，若一致 → 确认 encoder spatial output 是 axis（不论是通过 input res 还是 blur 实现）

---

### Training 12: Foveated σ=2 (`foveated_sigma2_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_sigma2_gibson` |
| GPU | A100 |
| Wall | ~3–4 days on A100 |

**Motivation**:  F1 *foveation strength* sweep, **low-blur endpoint**.
σ=2 is barely-foveated (peripheral degradation is mild).  Predicted to
behave like uniform — substitution as strong as the encoder allows.

**Specific prediction**:  R²(GPS\|h₂) in $[-0.1, +0.2]$ — at-floor like
uniform.

**Why it matters**:  Confirms that the foveation effect we observe at
σ=8 isn't an artifact of *some* peripheral blur — when blur is light, the
condition reverts to uniform-like substitution.

#### 中文 — 不符合预期 investigation 协议

预期：R² ∈ [-0.1, 0.2]（uniform-like floor）。

若 **R² > 0.4**（弱 blur 居然 preserve GPS code，异常）：
- 视觉检查 σ=2 输出（应非常接近 uniform）
- 若看上去和 uniform 一样，R² 却差很多 → 实现错误（可能 σ=2 实际应用时被 clip / 误用），debug source code
- 若视觉确实有差异 → 跑 seed=2 confirm；若一致 → 就算 σ=2 也足以触发 partial bottleneck，paper 改写 σ-sweep 描述
- ❌ 不要 paper integrate 单点 anomaly
- ✅ 实现 + seed=2 都 confirm 才动 paper

若 **R² ≈ 0**（符合预期）：
- 与 standard foveated R² ≈ 0 + uniform R² ≈ 0 一起，formed continuous floor → 支持 "Gaussian blur in this regime doesn't trigger bottleneck"

---

### Training 13: Foveated σ=12 (`foveated_sigma12_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_sigma12_gibson` |
| GPU | A100 |
| Wall | ~3–4 days on A100 |

**Motivation**:  F1c mid-strength.  Fills the σ ∈ {2, 8, 12, 20} sweep
between standard (σ=8) and strong (σ=20).  Useful for the F1c monotonicity
plot in App E.

**Specific prediction**:  R²(GPS\|h₂) in $[0.1, 0.4]$.

#### 中文 — 不符合预期 investigation 协议

预期：R² ∈ [0.1, 0.4]，介于 σ=8 和 σ=20 之间。

若 **σ-sweep 非 monotonic**（σ=12 R² 高于 σ=20 或低于 σ=8，破坏 σ → R² 单调）：
- 多 seed 验证（最少 σ=12 第二个 seed），看是否 seed-noise 导致
- 若仍 non-monotonic → blur strength 不是连续 axis，可能某个 σ 范围有 phase transition / encoder collapse
- ❌ 不要画 4 个点的 sweep figure 假装 monotonic（如把 σ=12 的 anomaly 隐藏在大 error bar 内）
- ✅ 报 raw 4 点 + 多 seed std；若没 monotonic，appendix discuss 可能的 phase transition + propose 补 σ=10/14/16 拉密 sweep

若 **R² ≈ 0**（提前到 floor）：
- 与 σ=8 一致；说明 σ=12 没多 informative，但不 contradict mechanism
- §App E 直接报 σ-sweep flat between σ=8..20 + log-polar 突变 → 强化 "blur 不够"

---

### Training 14: Foveated shifted (`foveated_shifted_gibson`) — H3 static control

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_shifted_gibson` |
| GPU | A100 |
| Wall | ~3–4 days on A100 |

**Motivation**:  Static gaze hardcoded at $(0.49, 0.62)$ rather than image
center $(0.5, 0.5)$.  This is the **H3 static control** — same architecture
as foveated (fix), only the gaze location changes.  It tests whether the
behaviour difference between foveated (fix) and uniform is *because* gaze
is at center, or *because* gaze is fixed.

**Specific prediction**:  R²(GPS\|h₂) within ±0.15 of standard foveated
(fix)'s ≈0; SPL within ±0.05 of foveated's 0.75.

**Why it matters**:  Pairs with stochastic gaze (Tier 1) for the H3
section — static-shifted vs.\ static-center vs.\ stochastic = three points
on the gaze-mobility axis.  Without this, we can't claim "gaze location"
and "gaze dynamics" are separable axes.

#### 中文 — 不符合预期 investigation 协议

预期：R²(GPS\|h₂) 与 standard foveated 相差 ≤ 0.15；SPL 相差 ≤ 0.05。

若 **shifted 与 standard 显著不同**（gaze 位置影响 H1）：
- 这意味着 gaze location 是 H1 的 secondary axis（不只是 dynamics 才影响）
- 跑 multi-position 验证：在 (0.3, 0.5), (0.7, 0.5), (0.5, 0.3), (0.5, 0.7) 各跑 1 次（这些是 4 个 cardinal shift；每跑 ~2 天）
- 若各位置都有差异 → gaze location 是 axis，§4.6 改写：H3 不只是 "dynamics"，也包括 "static location"
- ❌ 不要把 "shifted = standard, gaze location 不重要" 直接简单结论 → 单 shift 位置不足以 generalize
- ✅ multi-position 数据更 informative，§4.6 + appendix table 完整呈现

若 **shifted ≈ standard**（符合预期）：
- gaze location 对 H1 没影响，only dynamics 才会影响 → H3 narrative 正常 integrate
- §4.6 写 "static-foveated, static-shifted, stochastic 三点对比，前两者一致 R²≈0，stochastic R²=X" → clean three-point H3 figure

---

## Skip-list — do NOT run

| Config | Why skip |
|---|---|
| `foveated_sigma4_gibson` | Too close to σ=2 / σ=8; marginal information value. |
| `foveated_normaliser_gibson` | F2 normalizer ablation; already in App E from Izar runs. |

If Tier 3 finishes ahead of schedule and you have spare GPUs, ping wxu
before launching anything not on this list.

---

## Pre-flight checklist (before launching anything)

```bash
cd /path/to/cs503-project

# 1. Pull latest repo state
git pull origin main
# Latest commit at submission prep ≥ 9684d3e (or ask wxu for the head).

# 2. Verify the 12 configs exist
ls habitat_configs/ddppo_pointnav_{foveated_stochastic,matched32,matched64,matched96,matched128,matched192,foveated_logpolar,foveated_v2,foveated_strong,foveated_sigma2,foveated_sigma12,foveated_shifted}_gibson.yaml
# Should list 12 files.

# 3. Verify the policies are registered
python -c "
import sys; sys.path.insert(0, '/path/to/cs503-project')
import src.habitat
from habitat_baselines.common.baseline_registry import baseline_registry
for name in [
    'FoveatedStochasticGazePolicy',
    'FoveatedSigma2WijmansPolicy',
    'FoveatedSigma12WijmansPolicy',
    'FoveatedStrongWijmansPolicy',
    'FoveatedLogPolarWijmansPolicy',
    'FoveatedShiftedGazePolicy',
]:
    cls = baseline_registry.get_policy(name)
    print(f'{name}: {cls is not None}')
"
# All should print True.

# 4. Smoke test the new stochastic gaze policy (under 1 min)
python scripts/cluster/smoke_policy.py FoveatedStochasticGazePolicy
# Should print "policy forward pass ok" (or equivalent).

# 5. Dataset check (run this even if you ran the dataset setup yesterday)
ls $HABITAT_DATA/datasets/pointnav/gibson/v1/train_extra_large/content/ | wc -l   # Expect 411
ls $HABITAT_DATA/datasets/pointnav/mp3d/v1/train/content/ | wc -l                   # Expect 61
ls $HABITAT_DATA/datasets/pointnav/mp3d_gibson/v1/train/content/ | wc -l            # Expect 472
# If any number is wrong, see ../docs/DATASET_SETUP.md.
```

If any of (1)–(5) fails, **do not launch anything** — ping wxu first.

---

## How to ship trained checkpoints back to Izar

The friend's hc cluster has no shared filesystem with EPFL's Izar.  Shipping
is via **rsync over SSH using a deploy key** (one-time setup below).

### One-time setup (~5 minutes)

```bash
# On hc cluster, create a fresh deploy keypair
ssh-keygen -t ed25519 -f ~/.ssh/id_izar_wxu_deploy -N ''

# Send the public key (~80 chars one line) to wxu via Slack / email.
cat ~/.ssh/id_izar_wxu_deploy.pub
```

wxu adds it to Izar's `~/.ssh/authorized_keys`; friend confirms with:

```bash
ssh -i ~/.ssh/id_izar_wxu_deploy wxu@izar.epfl.ch 'echo connected; date'
```

### Per-run shipping

After each training hits 250M frames (or 8-day partial — whichever first):

```bash
bash scripts/cluster/ship_to_izar.sh <RUN_NAME>
# e.g.: bash scripts/cluster/ship_to_izar.sh foveated_stochastic_gibson
```

This rsyncs `latest.pth` + `ckpt.{10,20,30,40,49}.pth` + `tb/` (training
curves) into `/scratch/izar/wxu/habitat_checkpoints/<RUN_NAME>/`.

wxu's `probe_hc_arrival.sh` cron on Izar detects the new files within 30
minutes and auto-submits the probing pipeline.  No further action needed
from friend after running the ship script.

### Fallback if SSH outbound is blocked

(rare on academic clusters but possible)

```bash
# rclone with a shared Google Drive folder
rclone copy <ckpt-dir>/ shared:cs503_paper/<RUN_NAME>/
# wxu then pulls: rclone copy shared:cs503_paper/ /scratch/izar/wxu/habitat_checkpoints/
```

3.5 GB total transfer for all 14 runs — comfortable for any method.

---

## Daily check-in flow

| Frequency | Friend does | wxu does |
|---|---|---|
| Every morning | `squeue -u $USER` snapshot → Slack | Confirm crons clean on Izar |
| When training hits 250M | run `ship_to_izar.sh <run>` | wait for `probe_hc_arrival.sh` to detect |
| When ship script printed "ok" | confirm to wxu via Slack | run probing on landed ckpts |
| If anything weird | flag wxu BEFORE acting | diagnose |

The probing pipeline on Izar takes 20–40 min per run.  R² numbers + figs
update on the wxu side; friend doesn't need to look at them unless wxu
flags an unexpected result.

---

## Failure signatures — when to escalate

| Symptom | Likely cause | Action |
|---|---|---|
| `nan_sanitised > 0` in TB metrics | Numerical instability (rare on A100/H200; we patched the worst case). | Continue training; the fix absorbs it. |
| Stochastic gaze: σ → 0.05 (its lower bound) and pinned | PPO is suppressing exploration. | Continue 50M more frames; if still pinned, ping wxu. |
| Stochastic gaze: SPL < 0.5 by 50M frames | Too much gaze noise hurting navigation. | Continue but flag — may need to reduce σ_max. |
| Scaling sweep K=N converges to coarse-1×1 R² when expected to be lower | Encoder collapsing earlier than predicted. | Continue, this would actually *support* the substitution mechanism (good!) — flag for paper. |
| OUT_OF_MEMORY on A100 | Batch size / num_envs mismatch (A100 has 40GB or 80GB; some configs assumed H100's 80GB). | Reduce `num_environments` from 4 → 2 in sbatch wrapper, ping wxu. |
| Job killed by walltime | Just resubmit from latest ckpt. | The training picks up from `latest.pth`. |
| Probe-pipeline output (on Izar) stalls for 24h after ship | wxu's cron may have died. | Ping wxu. |

---

## Rough daily schedule

```
Day 1 (today)    Pre-flight checks. Launch Tier 1 (5 trainings; assign blind s2 to H200).
Day 2-3          Tier 1 training; first ckpts ship to Izar at ~125M frames.
Day 4-5          Tier 1 finishes (A100 ~3.5 days; blind on H200 ~3 days).
                 Ship final ckpts. Launch Tier 2 (5).
Day 6-7          Tier 2 training. wxu probes Tier 1 results.
Day 8-9          Tier 2 finishes. Skip Tier 3 (won't fit on A100 timeline).
Day 9 (2026-05-06) Final paper integration. Submit.
```

**Tier 3 reality check**: with 4× A100 + 1× H200 and ~3.5–4 days per
training, 14 trainings need 3 sequential rounds × 5 GPUs = ~10–12 days
to fully clear.  We have **9 days**.  So **Tier 3 likely won't fit** —
plan around Tier 1+2 (10 trainings) and only launch Tier 3 if Tier 1+2
finishes ahead of schedule, which requires H200 + lucky cluster
availability.

Submission target: clean NeurIPS submission with **multi-seed (3 conditions)
+ scaling sweep (5 K-points) + stochastic gaze H3 + falsifiable log-polar**.
That's the minimum credible set per our review-risk analysis.

---

## Quick links

- Dataset setup: [../docs/DATASET_SETUP.md](../docs/DATASET_SETUP.md)
- Paper TeX: [../docs/NeurIPS_2026/neurips_2026.tex](../docs/NeurIPS_2026/neurips_2026.tex)
- Sleep log (wxu's autonomous overnight progress): [../docs/NeurIPS_2026/SLEEP_LOG.md](../docs/NeurIPS_2026/SLEEP_LOG.md)
- One-time cluster setup helper: [setup.md](setup.md)

---

*— wxu, 2026-04-27.  If anything in this doc seems wrong / missing, ping
me on Slack before improvising.*
