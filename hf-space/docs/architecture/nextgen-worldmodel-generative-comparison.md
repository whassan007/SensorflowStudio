# Generative Architecture Comparison for Counterfactual Scene Generation

Companion to `sensorflow/nextgen/worldmodel.py`. Question: if/when the
deterministic scene transformer is outgrown, what class of generative world
model should back `ExternalWorldModelAdapter`?

The consumer contract is fixed and generator-agnostic: full ground truth out,
determinism per (source, recipe, seed), GENERATED provenance labels, and no
exemption from the validity gate (`nextgen/validity.py`).

## Options

**Diffusion world model** — denoising-based generation of future scene
states (or sensor frames) conditioned on scene context and edit masks.

**Transformer (autoregressive) world model** — tokenized scene/agent states
rolled out step by step conditioned on history and control/edit tokens
(the family most proprietary driving world models belong to).

**Hybrid** — transformer backbone for dynamics/agent interaction with a
diffusion decoder head for continuous state/appearance refinement.

## Comparison

| Criterion | Diffusion | Transformer WM | Hybrid |
|---|---|---|---|
| Fidelity (per-frame realism) | Highest; excels at appearance and texture | Good at dynamics, weaker at continuous detail | High on both, at integration cost |
| Controllability (recipe adherence: "this actor brakes at t=1.0s at 7.5 m/s^2") | Weak-to-moderate; edit conditioning is indirect (inpainting/guidance) | Strong; edits are just conditioning tokens; per-actor intervention is natural | Strong (inherits transformer conditioning) |
| Temporal/kinematic consistency | Failure mode: per-frame plausibility with cross-frame drift — exactly what `validity.check_temporal` and `check_identity` exist to catch | Strong; dynamics are the training objective | Strong |
| Latency / cost per scenario | High (many denoising steps × frames) | Moderate (one pass per step; KV-cached rollouts) | Highest |
| Determinism / reproducibility | Achievable (fixed noise, fixed sampler) but fragile across library versions | Straightforward (greedy/temperature-0 decode, seeded sampling) | Fragile at the diffusion head |
| Ground-truth availability | Hard if pixel-space (GT must be re-derived — a labeling problem again) | Natural if state-space (states ARE the GT) | State-space GT from backbone; decoder is presentation-only |
| Validation burden on our gate | Highest: every sample needs full 5-check gating; expect high rejection on temporal checks | Lowest of the learned options; sensor consistency (`check_sensor`) still mandatory | Medium |
| Fit to our transformation catalogue (actor edits, environment swaps, scene composition) | Better for environment/appearance edits (clear→fog) | Better for actor/behavior edits (sudden brake, crossing, emergence) | Covers both |

## Recommendation (pragmatic)

**State-space transformer world model first; diffusion only as an optional
appearance decoder; i.e. adopt the hybrid shape but build/buy it in that
order.** The reasons are specific to this platform: (1) our counterfactual
recipes are behavior edits far more often than appearance edits — 10 of the
15 transformations in `counterfactual.TRANSFORMATIONS` are actor/scene
kinematics, which transformer conditioning handles natively; (2) our
evaluation consumers need ground-truth states, not pixels —
`closedloop.py` and `validity.py` operate on `ActorTrack` kinematics, so a
state-space generator plugs into `ExternalWorldModelAdapter` without a new
labeling pipeline; (3) our gate's hardest checks (temporal continuity,
identity stability) are precisely diffusion's characteristic failure modes,
so a diffusion-first choice maximizes validation burden; (4) determinism —
required for lineage-valid launches (`lineage.py`) — is cheap for
temperature-0 transformer decoding and fragile for diffusion samplers.
Until any of that is bought or built, the deterministic transformer in
`counterfactual.DeterministicSceneTransformer` remains the live
implementation: zero fidelity risk, perfect controllability and
reproducibility, bounded expressiveness — the right trade for a platform
whose first duty is evidence integrity.
