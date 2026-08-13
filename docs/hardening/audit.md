# Sensorflow Studio — Production Hardening Audit

**Scope.** Full repository, with fixes restricted to the committed packages
(`sensorflow/{evaluation,megaeval,safety,seqeval,rca,raremine,bevfusion,vitis,metrics,schemas,adapters}`,
sensorflow root modules, `main.py`, `static/`) plus the new `sensorflow/hardening/` namespace.
`app_backend.py` was audited but is under a one-additive-edit budget (five agents in flight);
its findings are follow-ups. In-flight namespaces (`agentic`, `nextgen`, `hillclimb`, `retro`,
`studio_ux`) were read for context only.

**Machine-readable version:** [`audit.json`](audit.json) — the readiness scorecard and
`/api/hardening` endpoints are computed from it.

---

## 1. Findings table

Severity: **C**ritical / **H**igh / **M**edium / **L**ow. Full detail per finding in `audit.json`.

| ID | Area | Existing approach | Problem | Sev | Correct approach | Disposition |
|----|------|-------------------|---------|-----|------------------|-------------|
| F-001 | Mock as real | `app_backend.py:959-1096,804-821` — invented "CA Statewide SSAM dataset" with claimed FHWA provenance | Fabricated safety data served as a real federal dataset, no simulation marker | C | Provenance labels + versioned threshold config | Follow-up (edit budget); config shipped in `hardening/safety_config.py` |
| F-002 | Mock as real | `app_backend.py:492-581` — "Successfully ingested … from HF" over Unsplash stock photos | API affirmatively lies about ingestion; false lineage | C | `simulated` flag + provenance enum | Follow-up (edit budget) |
| F-003 | Temporal tracking | `sensorflow/temporal_tracker.py:120` — `abs(track.state[2] if len(...)>2 else 0 - yaw)` | Precedence bug: "yaw penalty" is actually `abs(vx)`; penalizes fast objects, ignores yaw, mixes units with the meter gate | C | Wrapped yaw difference vs last observed box | **Fixed now** + regression test |
| F-004 | Leakage | `sensorflow/perception_automator.py:83-104,167-173` + `quality_gate.py:38-56` | Proposals built FROM ground truth (fallback masks, classes copied by index), then benchmarked against the same GT → circular evaluation; export gate can pass on leaked info | C | Provenance-marked proposals; gate refuses GT-derived runs | Follow-up (behavioral contract); contracts shipped in `hardening/contracts.py`; determinism half fixed (F-008) |
| F-005 | Mock as real | `app_backend.py:234-237` — fabricated decreasing loss curve on parse failure | Diverging run renders as healthy convergence | C | Empty list + `parse_status` field | Follow-up (edit budget) |
| F-006 | Cache correctness | `sensorflow/megaeval/cube.py:106-124,229-231` — query cache keyed on (population, model, query) only | Key omits run identity (label_version, overrides, seed): runs with injected regressions collide with their baselines; wrong cached rows served | C | Key on immutable `run_id` (pins full lineage) | **Fixed now** + collision test |
| F-007 | Safety thresholds | `app_backend.py:1099-1111` vs `:1136-1140`; `rare_events.py:84-89` | Magic severity weights, caps, band cutoffs; TWO different weightings for the same severity concept across endpoints; TTC/PET literals buried | H | Single versioned registry with per-value provenance (FHWA vs ILLUSTRATIVE) | **Registry shipped** (`hardening/safety_config.py`); app_backend wiring is follow-up |
| F-008 | Determinism | `perception_automator.py:115-120` — unseeded `np.random.randn` fake LiDAR | Non-reproducible labeling; unlabeled simulation in production path | H | Seeded per-path RNG, explicit SYNTHETIC_FALLBACK label | **Fixed now** + determinism test |
| F-009 | Metric validity | `quality_gate.py:90-123` — boxes pooled across ALL frames before matching | Frame-1 prediction can match frame-50 GT; mAP/mAR/errors computed on impossible matches; launch gate consumes them | H | Per-frame matching, aggregate counts | **Fixed now** + cross-frame test |
| F-010 | Regression stats | `evaluation/regression.py:50-72` — point-delta vs fixed tolerance, 9 metrics, no n/CI/multiplicity | Noise flagged on small runs, real regressions ignored on huge runs; family-wise error uncontrolled | H | CI-aware decision (Wilson / seqeval) with tolerance as practical margin | **Fixed now (additive)**: optional `sample_sizes` → Wilson-CI decision; seqeval delegation documented |
| F-011 | Regression stats | `evaluation/pipeline.py:339-342` — every run becomes the next baseline | Rolling baseline masks slow drift (1pp/run forever undetected) | H | Pinned blessed baseline + cumulative drift alert | Follow-up (product decision) |
| F-012 | Grader independence | `evaluation/graders.py:170-231` — weighted-majority consensus, independent-error simulation, arbitrary weights | Correlated graders (shared backbone/data) counted as independent evidence; consensus overstated | H | Dependence matrix → effective grader count → adjusted confidence | **Layered fix**: `hardening/quality.py` (graders.py untouched) |
| F-013 | Security | `app_backend.py:80-104` — client-supplied arbitrary path read/write | Path traversal by design from unauthenticated HTTP | H | Allowlisted config dir | Follow-up (edit budget) |
| F-014 | Mock as real | `main.py:17-90,208-277` — Mock engines return constant risk metrics as `status: success`; row count as "processing seconds" | No simulation marker crosses the API boundary | H | Additive `simulated`/provenance markers; real elapsed time | **Fixed now** (additive fields) |
| F-015 | HITL routing | `evaluation/pipeline.py:376-399`, `mitl_copilot.py:40-66` — flat FIFO review batches; `pred_tracks[:3]` as "evidence" | No information-value prioritization; no acceptance metrics for routing itself | H | Risk×Uncertainty×Novelty×TrainingValue + Pareto alternative; HITL precision/critical-miss metrics | **Layered fix**: `hardening/hitl.py`; queue wiring follow-up |
| F-016 | Temporal tracking | `temporal_tracker.py:26-39` — Q=0.1·I, R=0.5·I | Uncalibrated noise; position and velocity share a scalar | M | White-noise-acceleration Q, sensor-spec R | Follow-up |
| F-017 | Scoring provenance | `quality_gate.py:158-169` — 40/20/20/20 quality score | Arbitrary headline scalar looks authoritative | M | Keep decomposed card authoritative; version the scalar | Follow-up |
| F-018 | Metric naming | `metrics/perception_3d.py:104-125` — "mAP" without ranking/PR curve | Misnomer vs literature mAP | M | Rename to precision@IoU or implement ranked AP | **Docstring fixed now**; rename is breaking → follow-up |
| F-019 | Temporal metrics | `metrics/temporal_mot.py:37-52` — nearest match with no distance gate | Phantom/hidden ID swaps from arbitrarily distant "matches" | M | Gate at 2.0 m (same as fragmentation) | **Fixed now** (default kwarg) |
| F-020 | Novelty scoring | `rare_events.py:152-165` — rarity=1−4·freq; confidence=0.55+0.4·rarity | Confidence fabricated FROM rarity — inverts evidence | M | Evidence-based confidence; real kNN/centroid novelty | Follow-up; corrected scorer in `hardening/sampling.py` |
| F-021 | Ensemble robustness | `detectors.py:379-385` — failed detector → silent zeros averaged in | Silent degradation masks anomalies | M | Record failures, exclude from ensemble | **Fixed now** |
| F-022 | Stats fallback | `vitis/hil.py:59-84` — paired-t on correlated frame means | SEM understated without sequence clustering (dormant: seqeval importable) | M | Cluster at sequence level | Follow-up |
| F-023 | Config hygiene | `mitl_copilot.py:11-14` + 3 app_backend sites — personal tailnet LLM endpoint hard-coded | Env-specific infra in source | M | Env-var override | **Fixed now** in mitl_copilot; app_backend follow-up |
| F-024 | Geometric validation | `evaluation/validation.py:129-134` — global ground plane z=0 | Meaningless penalty on slopes | M | Local ground estimate | Follow-up |
| F-025 | Security | `main.py:110-117` — CORS `*` + credentials | Disallowed combination | M | Origin allowlist | Follow-up |
| F-026 | Scalability | `records.py`, `megaeval/runs.py:363-371`, `vitis/store.py`, `app_backend.py:17-22` | No storage/compute seam; all stores single-node, in-process, lost on restart | M | Interface protocols with labeled LOCAL impls | **Layered fix**: `hardening/interfaces.py` |
| F-027 | Cache (secondary) | `megaeval/runs.py:434-437` — insertion-order artifact eviction, no checksums | Hot-run eviction; no integrity detection | L | LRU + checksums (`hardening/cache_manifest.py` design) | Follow-up |
| F-028 | Mock (mitigated) | `app_backend.py:663-724` — invented benchmark numbers, but labeled `fallback_mock` | Label good; numbers still quotable | L | Nulls + descriptions | Follow-up |
| F-029 | Sampling (verified) | `megaeval/sampling.py:100-111` | **Verified correct** HT-style reweighting; caveats: allocation rounding, reviewer error not deconvolved | L | Largest-remainder alloc; Rogan–Gladen | Follow-up |
| F-030 | Meta supervision | `evaluation/pipeline.py:217-220` — injected_errors as supervision (held-out) | Signal only exists for synthetic data; dormant by default | L | Guard behind synthetic check | Follow-up |

**Counts:** 6 Critical, 9 High, 11 Medium, 4 Low → 30 findings.
**Dispositions:** 7 fixed now, 4 partially fixed now, 3 fixed as layered capability, 16 deferred with reasons.

---

## 2. Critical findings (narrative)

1. **The worst code is where predicted: the legacy layer.** `app_backend.py` fabricates a
   statewide safety dataset with false FHWA provenance (F-001), fabricates HuggingFace ingestion
   over stock photos (F-002), and fabricates training loss curves (F-005). None of it is labeled.
   The rigorous packages (seqeval, megaeval, raremine) postdate this code and label their
   simulations scrupulously — the platform's honesty discipline improved over time but was never
   retrofitted.
2. **A real math bug corrupts tracking** (F-003): the association "yaw penalty" is `abs(vx)` due
   to operator precedence. Every consumer of `TemporalTracker` inherited it, including
   `bevfusion.BEVMaskletTracker` (which overrides `_associate`, so it dodged the bug — likely why
   nobody noticed).
3. **Legacy quality metrics are computed on physically impossible matches** (F-009): pooling boxes
   across frames lets predictions match ground truth from other timestamps, and the launch gate
   consumes the result.
4. **The legacy auto-labeler grades its own homework** (F-004): without SAM, proposals are
   *derived from ground truth* then *scored against that ground truth*. Combined with F-009 this
   makes the legacy 3D pipeline's quality gate close to vacuous.
5. **The megaeval query cache can serve the wrong run's results** (F-006): the key ignores
   overrides/label version/seed, so an A-vs-B regression comparison can silently read A's numbers
   for B. Fixed by keying on `run_id`.
6. **Naive regression flags** (F-010, F-011): point-delta-vs-tolerance with a rolling baseline is
   exactly the pattern seqeval exists to replace — noise-flagging on small n, blind on large n,
   drift-masking across runs.

## 3. Architecture corrections

- **Provenance is the missing spine.** The codebase blurs MODEL_PREDICTION / AUTO_LABEL /
  VLM_INFERENCE / HUMAN_LABEL / CERTIFIED_GROUND_TRUTH in several places (F-002, F-004;
  `evaluation.records.GroundTruthType` is a partial version). `hardening/contracts.py` defines the
  `LabelProvenance` enum and pydantic contracts with mandatory provenance fields, plus adapters
  from existing records. Adoption path: new code imports contracts; existing packages are mapped
  via adapters until their next breaking release.
- **Storage/compute seam** (F-026): `hardening/interfaces.py` defines VectorDB, ObjectStorage,
  DistributedCompute, GPUInference, FeatureCache, MetadataStore, ExperimentTracking, Observability
  protocols; every current implementation is registered as labeled LOCAL/MOCK. megaeval's
  partition/reduce design already fits DistributedCompute cleanly.
- **Feature-cache manifests** (F-006, F-027): `hardening/cache_manifest.py` specifies content-
  addressed keys over *all* dependencies (data hash, model/label/evaluator/metric versions, config,
  precision flags), integrity checksums, and invalidation/eviction policy, with a migration note
  for the three audited caches (megaeval query cache, megaeval artifact cache, vitis reuse).

## 4. Statistical corrections

- Regression decisions on proportions now support Wilson-CI gating (F-010, implemented) and should
  fully delegate sequential use to `seqeval.PairedSequentialTest` (documented; hil.py already does).
- Evaluation sizing must be derived, not fixed: `hardening/power.py` computes required n from
  baseline rate, MDE, α, power, prevalence and pairing/cluster correlation (design effect from
  `rca.stats`), and sizes Tier 0-3 evaluations from those inputs.
- Consensus must be dependence-adjusted (F-012, implemented as a layer).
- Multiple-testing: seqeval ships e-BH; regression.py's 9-metric sweep and any future
  multi-metric gate should adopt it (documented follow-up).

## 5. Safety corrections

- Every safety threshold gets provenance: `hardening/safety_config.py` centralizes the SSAM
  severity weights/caps/bands, legacy alternate weights, rare-event TTC/PET literals and quality-
  gate thresholds, each tagged `FHWA_SSAM_DEFAULT` or `ILLUSTRATIVE_THRESHOLD` with a version.
- Severity must be consistent across endpoints (F-007): the two conflicting weightings are now both
  captured in the registry with the discrepancy documented; the wiring fix is a follow-up.
- `safety/ssam_ext.py` is the model to follow and required no fixes.

## 6. Code corrections (implemented now)

| Fix | File | Test |
|-----|------|------|
| Yaw-penalty precedence bug → wrapped yaw diff | `sensorflow/temporal_tracker.py` | `tests/test_hardening/test_surgical_fixes.py` |
| Seeded, labeled synthetic LiDAR fallback | `sensorflow/perception_automator.py` | same |
| Run-scoped query-cache key | `sensorflow/megaeval/cube.py` | same |
| Per-frame quality-gate matching | `sensorflow/quality_gate.py` | same |
| Wilson-CI regression option (`sample_sizes`) | `sensorflow/evaluation/regression.py` | same |
| Distance-gated ID-swap matching | `sensorflow/metrics/temporal_mot.py` | same |
| Ensemble failure recording/exclusion | `sensorflow/evaluation/detectors.py` | same |
| Env-overridable LLM endpoints | `sensorflow/mitl_copilot.py` | same |
| Simulation markers + real elapsed time | `main.py` | same |
| Truthful compute_map_mar docstring | `sensorflow/metrics/perception_3d.py` | n/a (docs) |

All fixes are API-preserving (signatures unchanged; new parameters optional with
behavior-preserving defaults, except where the previous behavior was itself the bug).

## 7. Production gaps

- Single-process, in-memory/JSON state everywhere (F-026); server restart loses training state.
- No authn/authz on any endpoint; arbitrary path read/write (F-013); CORS misconfiguration (F-025).
- No observability seam (metrics/traces); `interfaces.py` defines the protocol.
- Exabyte narrative vs single-node reality: megaeval's *shape* scales; its execution substrate does not (honest in code comments, absent in interfaces — now defined).

## 8. Prioritized remediation plan

**Fix now (done in this pass)** — F-003, F-006, F-008, F-009, F-010(partial), F-014, F-019,
F-021, F-023(partial), F-018(docs) + layered capabilities for F-007, F-012, F-015, F-026.

**Follow-up, ordered:**
1. **F-001/F-002/F-005/F-013 (Critical/High, S effort each):** label or remove fabricated data in
   `app_backend.py`; path allowlist. Blocked only by the concurrent-edit freeze — first change
   after unfreeze, wiring `hardening/safety_config.py`.
2. **F-004 (Critical, M):** provenance-marked proposals; gate refuses GT-derived runs. Requires
   coordinated re-baselining with in-flight consumers of the legacy pipeline.
3. **F-011 (High, M):** pinned blessed baseline + cumulative drift alert in labeleval pipeline.
4. **F-012/F-015 wiring (High, M):** route labeleval review queue through `hardening/hitl.py`;
   surface dependence-adjusted consensus in the quality UI.
5. **F-010 completion (High, M):** pipeline passes per-metric counts; adopt seqeval e-BH for the
   9-metric family.
6. **F-020 (Medium, S):** evidence-based rare-event confidence; adopt `hardening/sampling.py` novelty.
7. **F-016/F-017/F-024 (Medium, S-M):** calibrated Kalman noise; documented score weights; local ground plane.
8. **F-025/F-027/F-028/F-029/F-030 (Low/Medium, S):** hygiene items as listed.

## 9. Strengths (verified, to be reused — not reinvented)

- **seqeval** — anytime-valid e-process testing, cluster-mean correlation handling, e-BH
  multiplicity, three-outcome decisions. The statistical backbone; everything else should delegate.
- **megaeval** — correct stratified estimation with true-weight recombination (verified, F-029) and
  complete per-run lineage; aggregate-first cube with honest scan fallback.
- **raremine** — enforceable leakage guard (LeakageError + audited governance overrides) and
  hard modality discipline. Verified that evaluation/synthetic and megaeval paths cannot bypass it
  (they never construct raremine destinations); the guard pattern should be adopted platform-wide
  via contracts.
- **rca.stats** — cluster-robust paired deltas, ICC/design effect, PSI with practical labels.
- **safety/ssam_ext** — provenance-commented FHWA thresholds, honest simulation markers, real
  conflict geometry.
- **vitis** — "modeled_not_measured" discipline; explicit hardware stub; seqeval-first verdicts.
- **evaluation triage/detectors/grader-statistics** — explainable per-gate decisions; genuine
  seeded detector implementations; kappa statistics used in their correct regimes.
