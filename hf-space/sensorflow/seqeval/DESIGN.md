# seqeval — Efficient Sequential Regression Detection: Technical Design

**Goal.** Detect small model regressions (~2 percentage points on specific
classes/conditions) on every model update WITHOUT evaluating the full corpus,
with statistical validity under (a) sequential looks, (b) clustered
(non-independent) frames, (c) stratified non-proportional sampling, and
(d) many simultaneous hypotheses.

This document records the design *and the judgment calls*: which candidate
techniques were adopted, which were rejected, and why. The reference
implementation lives in this package and is exercised by `tests/test_seqeval/`.

---

## 1. Problem formulation

For each metric m (recall is the driver metric in v1) and each population
stratum h, let Δ_h = m_candidate(h) − m_baseline(h) measured on the same data.

With a **practical-significance margin** δ (default 0.005 = 0.5pp):

```
H0 (no material regression):  Δ_h ≥ −δ
H1 (material regression):     Δ_h < −δ
```

plus the separate equivalence-style claim used for PASS: reject `Δ_h ≤ −δ`
in favor of `Δ_h > −δ`.

**Absolute vs relative deltas.** The default driver is **absolute percentage
points**, for three reasons:

1. Safety cost is absolute: a 2pp recall drop means the same number of extra
   missed pedestrians whether baseline recall was 0.98 or 0.80. Relative
   deltas overstate changes on strong baselines (0.98→0.96 is "only −2%
   relative") and understate them on weak ones.
2. Absolute deltas keep Bernoulli variance algebra and margin semantics simple
   and uniform across strata; relative margins make the null hypothesis depend
   on the (estimated) baseline value, contaminating the test with baseline
   estimation error.
3. Absolute margins are what promotion policies elsewhere in this platform
   already use (`megaeval.analysis.DEFAULT_PROMOTION_POLICY` is expressed in
   absolute drops), so the gate composes with existing policy language.

Relative deltas are still *reported* in every evidence record (`rel_delta`)
because they are useful for human interpretation, especially for rare classes
with weak baselines. They are never the decision variable.

**Practical vs statistical significance.** With ~320k objects, a 0.05pp drop
can be "statistically significant" yet operationally meaningless. Hence the
margin δ: nothing is declared REGRESSION unless the evidence shows the drop
exceeds δ, and nothing is declared PASS unless the evidence shows any drop is
smaller than δ. Statistical significance against a point null of exactly zero
is deliberately not used anywhere.

## 2. The statistical unit: clusters, not frames

Frames/objects are not independent: objects in one container (scene/track)
share weather, lighting, scenario, scene-level hardness and — critically —
correlated model failures (one bad scene produces many misses). Treating
objects as iid understates variance by the design effect

```
DEFF = 1 + (m̄ − 1) · ICC        (m̄ = size-weighted mean cluster size)
```

and every "significant" result computed that way is anti-conservative.

Adopted design:

* **Inference unit = cluster**: all sampled objects of a stratum in one
  container. Per-cluster mean paired differences (bounded in [−1, 1]) are the
  observations fed to the confidence sequences and e-processes. This is valid
  under *arbitrary* within-cluster correlation — no correlation model is
  assumed — and whole clusters are sampled so the unit is never split
  (`planner.py` selects container groups, not objects).
* **Diagnostics, not corrections**: ICC (one-way ANOVA estimator), DEFF and
  n_eff = n/DEFF are computed per node (`units.py`) and attached to every
  evidence record. They are reporting artifacts; validity does not depend on
  them because clustering is handled structurally.
* An interesting empirical note the machinery surfaces: pairing itself removes
  most cluster correlation (the shared scene effect hits both models and
  cancels in d_i), so ICC on paired differences is small (~0.07 in the worked
  example, DEFF ≈ 1.2) even when ICC on raw outcomes is large. We keep the
  cluster unit anyway — the residual correlation is real and free to handle.
* **Clustered bootstrap** (resampling clusters, not objects) is the prescribed
  fallback for metrics that don't decompose into per-object outcomes (§3). It
  is not needed for the v1 recall driver.

Rejected: variance-inflation-only corrections (divide n by an assumed DEFF≈2
and pretend units are iid). Rejected because ICC varies by stratum and model
pair; a fixed inflation is either wasteful or invalid, and it composes poorly
with anytime-valid methods whose guarantees are per-observation.

## 3. Paired evaluation as the core design

Both models are evaluated on the **same sampled units** (`paired.py`), and all
inference is on the per-unit difference d_i = c_i − b_i ∈ {−1, 0, +1}.

Why this is the single biggest efficiency lever: for two independent samples,
Var(m̂_c − m̂_b) ≈ 2·p(1−p)/n (~0.18/n at p=0.9). For paired evaluation,
Var(d) = ν − Δ², where ν = P(models disagree) — the discordance rate. Adjacent
model versions agree on almost everything (measured ν ≈ 0.03–0.06 in the
harness), so pairing cuts required samples by roughly 3–10×. The paired
design also removes the between-unit difficulty variance *and* the shared
scene effects (§2).

**McNemar correspondence.** For binary per-object outcomes the information in
the paired design is exactly the discordant counts: n01 (baseline right,
candidate wrong) vs n10 (the reverse) — McNemar's setup. We do not use the
classical McNemar *test* (it is fixed-n); instead the same discordant-pair
information drives (a) the sequential e-process on d_i and (b) the
Beta-Binomial posterior of §14, which is literally a Bayesian McNemar. The
counts n01/n10 are reported in attribution because they are the most
interpretable summary of "what changed".

**Which metrics get which treatment:**

| Metric | Per-object outcome? | Sequential treatment |
|---|---|---|
| recall (per-GT-object TP/FN) | yes, Bernoulli | confidence sequence / e-process on paired d_i ✔ (v1 driver) |
| precision (per-prediction) | yes, but the unit set differs per model (each model's own predictions); pair at the cluster level | CS on cluster-level paired precision differences |
| safety_recall, per-stratum recall | yes | same as recall ✔ |
| mean IoU (matched pairs) | bounded continuous per object | CS on paired IoU differences (empirical-Bernstein handles [0,1] naturally) |
| AP / mAP / AUC | **no** — rank statistics over the whole set; not a mean of per-object terms | **paired clustered bootstrap** at checkpoints (resample clusters, recompute both models' AP on the same resample, CI on the difference). No anytime-valid guarantee: schedule at *pre-registered* checkpoints (end of each budget stage) with alpha-spending across the (few, fixed) looks |

This split is a deliberate honesty boundary: Bernoulli-style anytime-valid
treatment is used exactly where a metric is a mean of bounded per-unit terms;
composite rank metrics get bootstrap at a small number of pre-committed looks
rather than pretending they support continuous monitoring.

## 4. Sampling design: hybrid Neyman + risk floors + safety minimums

Candidates considered:

* **Proportional allocation** — matches the population, but starves exactly
  the strata we care about (pedestrian|night is ~5% of objects; a proportional
  10k sample gives ~500 units, hopeless for 2pp detection). Rejected as the
  primary rule.
* **Equal allocation** — maximal per-stratum power for fixed budget, but
  wastes budget on huge easy strata and makes population-level estimates
  needlessly noisy. Rejected.
* **Risk-weighted only** (as in `megaeval.sampling`) — right instinct, no
  optimality story; can be arbitrarily far from variance-optimal. Rejected as
  the *sole* rule, kept as a multiplier.
* **Neyman allocation** n_h ∝ N_h·σ_h — variance-optimal for the *population*
  estimate, but per-stratum guarantees are incidental, and σ_h must not come
  from candidate outcomes (bias firewall, §13). Adopted as the baseline using
  a candidate-independent PRIOR variance profile (documented rates by
  class/condition; a pilot baseline-only estimate would also qualify).

**Adopted hybrid** (`planner.py`), in order: Neyman shares from prior σ_h →
multiplicative risk weights (VRU ×1.5, night ×1.3) → mandatory minimum per
stratum (200) → pre-registered safety-critical strata get a floor sized for
the target MDE (2500 by default, §10) → truncate at N_h, select whole
clusters in a seeded shuffled order.

**Unbiasedness under oversampling.** Every sampled object carries the
Horvitz–Thompson weight w_h = N_h/n_h. Population and class-level estimators
are weight-corrected (weighted cluster means feed the overall/class/difficulty
nodes), so the classic stratified identity `E[Σ_h W_h p̂_h] = p` holds and
oversampling safety strata does not bias headline estimates. Per-stratum
(leaf) inference is self-weighting and needs no correction.

## 5. Multi-stage pipeline

```
sanity (≈500 units)  →  stratified screening (~15% of plan)
    →  sequential confirmation (batched to 100% of plan, anytime-valid)
    →  targeted escalation (reserve clusters, suspect/safety strata only)
```

* **Sanity**: fingerprints resolved, frozen plan hashed, smoke batch checks
  for degenerate outputs (rate ∉ (0.2, 1], discordance ≥ 0.5 ⇒ abort — a
  broken export should fail in seconds, not consume the budget).
* **Screening** flags "suspect" strata early (e-value ≥ 1.5 — soft evidence,
  *never* a decision) to prioritize escalation later. Screening consumes the
  same frozen plan; it does not select units based on outcomes (§13).
* **Sequential confirmation** processes the remaining plan in batches; all
  decisions (REGRESSION / PASS) can fire at any batch boundary because the
  underlying processes are anytime-valid.
* **Escalation** pulls from each stratum's pre-frozen reserve (shuffled at
  plan time) for strata still undecided that are suspect or safety-primary,
  up to a hard cap. Escalation to *full* evaluation of a stratum is the
  terminal step of the same mechanism (the reserve is the rest of the
  stratum).

## 6. Sequential methodology: anytime-valid confidence sequences, not CUSUM

**Why repeated fixed-sample CIs are wrong.** A 95% CI computed at n has 5%
miscoverage *at that single n*. Recompute after every batch and stop on the
first exclusion of the null and the type-I error inflates monotonically —
by the law of the iterated logarithm the statistic crosses any fixed-n
boundary infinitely often under the null, so "peek until significant" has
error probability → 1. Evaluation gates that check after every batch are
exactly this setting.

**Adopted: confidence sequences / e-processes** (`sequential.py`):

* Empirical-Bernstein confidence sequence (Waudby-Smith & Ramdas 2023,
  predictable plug-in): a CS valid *simultaneously over all n*, so the gate
  may look after every batch and stop whenever it likes. Empirical-Bernstein
  rather than Hoeffding because paired differences have tiny variance
  (ν ≪ 1/4) and EB width scales with the realized variance; a Hoeffding CS
  would be ~3–4× wider here (Hoeffding rejected for power, kept as a mental
  baseline).
* One-sided **betting e-processes** for the two directed nulls. The wealth
  process K_t = Π(1 − λ_t(X_t − m)) is a nonnegative supermartingale under
  H0: μ ≥ m for any predictable λ_t in the admissible range; Ville's
  inequality gives P(∃t: K_t ≥ 1/α) ≤ α. Bet sizing = max(empirical-Bernstein
  recipe, aGRAPA/approximate-Kelly λ ≈ drift/(σ̂² + drift²)), clipped —
  sizing affects power only, never validity. E-values (not just CIs) are kept
  because they compose: they feed e-BH in the hierarchy (§8) and multiply
  across independent looks/runs if evidence is ever pooled.
* Batch-incremental updates, sticky decisions, per-batch trajectory recording
  (`log e vs n` with the 1/α boundary) for the dashboard.

**Where CUSUM legitimately fits — and why not here.** CUSUM/Page tests are
designed for *change-point detection in an ongoing stream*: an unknown time
at which a monitored process shifts, with guarantees phrased as average run
length (ARL) to false alarm and detection delay. That is the right tool for
**production drift monitoring** (a deployed model's live metric degrading at
an unknown time — a reasonable future addition next to
`sensorflow/evaluation/regression.py`). It is the wrong tool for **gated A/B
evaluation**, where there is no change-point: the candidate is a fixed,
already-different model, the hypothesis is about a fixed mean difference, ARL
semantics don't map to per-release type-I error, and CUSUM provides neither
confidence intervals for effect size nor e-values for hierarchical multiple
testing. CUSUM is therefore explicitly rejected for the gate.
(Group-sequential designs — O'Brien-Fleming etc. — were also considered:
valid, but they require pre-committing to a fixed number of looks and a
maximum n, which fights the budget-staged, stop-any-time controller; the
anytime-valid framework strictly dominates for this use.)

## 7. Mandatory three-outcome decisions

Non-detection is NEVER equivalence. Each node ends in exactly one of:

* **REGRESSION** — the regression e-process crossed 1/α_allocated: confirmed
  drop beyond the margin. Gate: block.
* **PASS** — the equivalence e-process crossed 1/α_pass: confirmed that any
  drop is smaller than the margin (this is a one-sided equivalence /
  non-inferiority claim, the sequential analogue of a TOST bound). Gate:
  allow, but only when the overall node AND every pre-registered safety
  primary individually PASS.
* **INSUFFICIENT_EVIDENCE** — budget exhausted first. Gate: expand within
  budget if possible, else report with explicit language: *"not proven
  equivalent — do not treat as a pass."* (`controller.INSUFFICIENT_LANGUAGE`
  is exactly that sentence; the tiny-budget test asserts it.)

## 8. Hierarchical multiple testing

Testing ~28 nodes (overall, 6 classes, 18 class×condition strata, 3 difficulty
bands) at α each would give a family error of ~1−0.95²⁸ ≈ 76%. Naive
Bonferroni over everything destroys power for the one node that matters.
Adopted structure (`hierarchy.py`):

```
Level 1  overall                        α·0.30   single test
Safety   pre-registered primaries       α·0.30   split equally, tested individually
Level 2  class                          α·0.15   e-BH within level
Level 3  class × condition strata       α·0.15   e-BH within level (non-safety)
Level 4  difficulty bands               α·0.10   e-BH within level
```

Budgets sum to α, so the union bound gives family-wise control across levels;
within a level, **e-BH** (BH on e-values) controls FDR *under arbitrary
dependence* — mandatory here because levels re-aggregate the same units and
strata share scenes. e-BH composes with optional stopping because an
e-process value at any stopping time is a valid e-value (this is the reason
the implementation carries e-values and not just CSs).

**Gatekeeping and the masking trap.** Classic top-down gatekeeping ("test
children only where the parent rejected") was **rejected** for the regression
direction: a 2pp pedestrian-night drop inside a globally *improved* model is
invisible at level 1, and a gate that only descends on level-1 rejection would
never see it. Instead every level has its own reserved budget, and
**safety-critical strata are pre-registered primary hypotheses** with
dedicated alpha, tested individually and exempt from e-BH masking by
better-behaved siblings. The power test in the suite is exactly this case:
overall metric up ~0.3pp, pedestrian-night down 2pp — and the stratum node
must fire. Alpha propagation is top-down only in the *reporting* sense
(attribution descends into difficulty bands of regressed strata); decisions
never require a parent rejection.

## 9. Rare-class policy

* **Minimum counts**: every non-empty stratum gets ≥ 200 sampled objects
  (or all of N_h if smaller) — below that, even 10pp effects are undetectable
  and the honest answer is INSUFFICIENT_EVIDENCE, which the three-outcome
  logic produces naturally.
* **Oversampling with weight correction**: rare + risky strata get risk
  multipliers and floors (§4); HT weights keep population estimates unbiased.
* **Per-class thresholds**: the margin δ and alpha allocation are policy
  fields; safety primaries effectively run at tighter per-node alpha with far
  larger minimum samples. (A per-stratum δ_h map is a straightforward policy
  extension; v1 keeps a single δ for legibility.)
* **Escalation**: rare strata that end INSUFFICIENT but suspect consume their
  reserve up to the whole stratum. If a stratum is exhausted and still
  undecided, the ledger says so explicitly (stopping_reason
  `escalation_exhausted`) — the decision is a visible budget/detectability
  statement, not silence.

## 10. Sample sizes and power (with the design effect)

Fixed-n planning numbers for a Bernoulli recall metric at baseline p₀ = 0.9,
one-sided α = 0.05, power 0.9 ((z₀.₉₅+z₀.₉)² ≈ 8.57). Paired variance uses
Var(d) ≈ ν − Δ² with measured-typical discordance ν ≈ 0.04 + Δ:

| detectable Δ | unpaired, per model arm | paired, iid units | paired, DEFF = 2 | paired, DEFF = 2, anytime (~×1.35) |
|---:|---:|---:|---:|---:|
| 1pp  | 16,100 | 4,280 | 8,550 | ~11,500 |
| 2pp  |  4,150 | 1,280 | 2,550 |  ~3,450 |
| 5pp  |    745 |   300 |   600 |    ~810 |
| 10pp |    214 |   111 |   223 |    ~300 |

Notes: (i) pairing beats doubling the corpus — the unpaired column is per
*arm*; (ii) the anytime-valid overhead (~30–40% at these scales, shrinking
with the e-process's adaptive bets) buys the right to stop at any time, which
in expectation *saves* samples whenever the effect is larger than the MDE
(the 10pp test stops ~15× before budget); (iii) measured DEFF on paired
differences was ≈1.2, not 2 — the table's DEFF=2 column is the conservative
planning figure requested, the ledger reports the realized one. The default
safety floor (2,500) is the 2pp row with the realized design effect plus
head-room.

## 11. Compute-budget staging

Default policy on a 320k corpus: plan ≈ 25k objects, staged
`[15%, 40%, 100%]` of plan (≈ 3.8k → 10k → 25k) + escalation reserve, i.e. a
10K→50K→250K→full ladder scaled to plan size. Advancement rules: advance
while any node is undecided and budget remains; stop immediately on any
confirmed REGRESSION (sticky, anytime-valid) or when overall + all safety
primaries PASS with nothing undecided; escalate only suspect/safety strata;
terminal state otherwise INSUFFICIENT_EVIDENCE with per-stratum sample
accounting in the ledger.

## 12. Prediction caching

Baseline predictions are computed once per (dataset fingerprint, model
fingerprint) and reused across every candidate update
(`paired.PredictionCache`, disk + memory). Fingerprints cover everything
predictions depend on — dataset identity/seed/size, model version + effects
config — so **invalidation is by construction**: any change produces a new
key; nothing stale can ever be served. A compute counter proves in tests that
the second candidate evaluation performs zero baseline recomputation. In a
real deployment the fingerprint inputs become dataset content hash + model
checkpoint digest + inference config (thresholds, NMS, quantization), and the
cache stores per-unit predictions rather than a success vector.

## 13. Sampling-bias prevention

* The plan is built by a function that **cannot see candidate outcomes** (no
  such parameter exists), from population structure + a prior variance
  profile only, then persisted with seed, config and a content hash over the
  selected and reserve unit ids. The hash is recorded in every run's lineage;
  two candidates with opposite outcomes provably receive byte-identical plans
  (asserted in tests).
* Escalation draws from a reserve whose order was frozen at plan time, so
  even adaptive *budget* decisions never become adaptive *selection*.
* **If outcome-adaptive sampling is ever added** (e.g. active sampling of
  high-disagreement regions), estimates must be corrected with
  inverse-propensity weights w_i = 1/π_i under recorded, everywhere-positive
  selection probabilities π_i, and CS validity must be re-established for the
  weighted observations (bounded w_i·d_i, e.g. via truncated propensities).
  Without that correction, adaptive selection biases Δ̂ toward whatever the
  selector chases. v1 deliberately does not do adaptive selection.

## 14. Bayesian complement

Alongside every frequentist decision, `sequential.py` computes a Beta-Binomial
posterior P(Δ < −δ | data) from the discordant pairs (θ = P(regression pair |
discordant) with Beta(1,1) prior; Δ = ν(1−2θ)). Operational comparison:

* *For* the posterior: a single monotone number ("99.99% probable this is a
  real regression") that program managers read correctly without training;
  natural for a decision-theoretic cost tradeoff.
* *Against* (as the gate): posteriors have no frequentist stopping guarantee
  under continuous monitoring without committing to a prior + loss and
  accepting their audit burden; e-values give the same "accumulating
  evidence" narrative *with* an error guarantee any auditor can check
  (e ≥ 1/α), and they compose across strata (e-BH) and across looks by
  multiplication.

Verdict: **e-values gate, posterior explains.** Both are in the ledger.

## 15. Failure modes and mitigations

| Failure mode | Mitigation in this design |
|---|---|
| Peeking inflates type-I | everything anytime-valid; fixed-n intervals never gate (§6) |
| Correlated frames overstate n | cluster inference units; DEFF/n_eff in every record (§2) |
| Masking: overall improves, stratum regresses | reserved per-level alpha + pre-registered safety primaries (§8) |
| Multiplicity across strata | hierarchical budgets + e-BH within level (§8) |
| "No signal" read as "safe" | three-outcome logic; INSUFFICIENT carries "not proven equivalent" (§7) |
| Outcome-dependent sampling bias | frozen hashed plans, frozen reserves, no adaptive selection (§13) |
| Rare strata silently undecidable | floors, oversampling+weights, escalation, explicit exhaustion reason (§9) |
| Stale baseline predictions | fingerprint-keyed cache, invalidation by construction (§12) |
| Broken candidate export burns budget | sanity smoke stage aborts in ~500 units (§5) |
| Trivial-magnitude "significance" | practical margin δ in both null hypotheses (§1) |
| Irreproducible decisions | seeds + fingerprints + plan hash + config in lineage; end-to-end determinism asserted in tests |
| Metric doesn't decompose per object (mAP) | pre-registered checkpoint paired clustered bootstrap, not fake sequential (§3) |

## 16. Final recommendation (defaults shipped in `controller.DEFAULT_POLICY`)

Paired, cluster-unit, stratified evaluation against a frozen hybrid
Neyman/risk-floor plan; empirical-Bernstein confidence sequences + betting
e-processes with a 0.5pp practical margin; three-outcome decisions; hierarchical
alpha budget (30% overall / 30% safety primaries / 15% class / 15% stratum /
10% difficulty) with e-BH within levels; staged budget 15%→40%→100% of a ~25k
plan with targeted escalation; fingerprint-keyed baseline cache; Beta-Binomial
posterior as explanatory complement. CUSUM reserved for future production
drift monitoring, not release gating.

## 17. Worked example: planted 2pp pedestrian-night regression (real run)

Setup: population `320k` objects / 26,666 containers (megaeval synthetic,
seed 42). Baseline `model-v41`. Candidate `model-v42` with planted effects
{pedestrian|night: −2pp, global: +0.4pp} — the masking case. Default policy
(δ = 0.5pp, α = 0.05, plan target 24k). Ground truth measured on the full
corpus afterwards: overall Δ = **+0.28pp** (candidate looks *better*
headline-wise), pedestrian|night Δ = **−1.63pp** realized (N = 17,561,
baseline recall 0.801).

Run (deterministic, reproducible from lineage):

* Plan: 25,699 objects frozen across 18 strata; pedestrian|night allocated
  2,501 (floor-driven; weight w = 7.02) + 15,060-object reserve.
* Decision: **REGRESSION** at 26,705 objects total = **8.4% of the corpus**
  (~12× cheaper than a full paired evaluation), stopping reason
  `regression_confirmed`, during escalation of the suspect safety primary.
* Pedestrian|night node at stop: n = 3,507 objects in 1,195 clusters
  (ICC 0.065, DEFF 1.20, n_eff 2,923); Δ̂ = −1.94pp,
  95% CS [−3.97pp, −0.48pp]; discordant pairs n01 = 110 vs n10 = 42;
  log e-value **7.50** vs boundary log(1/α_safety) = **4.89**
  (e ≈ 1,808 ≥ 133); Beta-Binomial P(Δ < −δ | data) = 0.99999.
* Evidence trajectory (n, log e): (129, 1.5) → (1002, 1.6) → (2501, 2.8) →
  (3006, 4.9) → (3507, 7.5) — the dashboard plots exactly this against the
  4.89 boundary.
* Overall node: Δ̂ = +0.03pp, CS [−0.51pp, +0.38pp] — correctly *not*
  regressed and *not yet* provably equivalent; the hierarchy is what surfaced
  the stratum despite the healthy headline.
* Ledger: 28 node records, each with the full §15-proof field set; lineage
  carries plan hash, fingerprints, seed and the entire statistical config.

The same engine with a null candidate (no planted effect) across 24 Monte
Carlo repetitions produced **zero** false REGRESSION verdicts (α = 0.05), and
with a planted 10pp effect stopped at ~2.8k samples — 11% of plan, <1% of the
corpus (both asserted in `tests/test_seqeval/`).
