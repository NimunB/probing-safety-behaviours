# Major decisions — what was chosen, why, and where it was decided

The author's decision record, compiled 2026-07-30 from `safety_probes_plan.md` (entries **#N** /
**M-N**) and the conversation transcripts in `claude-conversations/` (cited as `file Lnnn`;
`7018fd8b` = the full 07-10→07-20 session, `c8aa4b41` = the 07-20→07-28 session). Newest thinking
wins where entries superseded each other. Companion: `thesis_walkthrough.md` §5 (compact ledger),
`claude-conversations/summary.md` (the confusion/learning review).

---

## Provenance at a glance — every probing/steering decision, and whose it is

Source tags: **[GoT]** Marks & Tegmark · **[Apollo]** Goldowsky-Dill et al. · **[HS]** McKenzie
et al. · **[CAA]** Rimsky et al. · **[vdW]** van der Weij et al. · **[NB]** the ARENA
1.3.1_Linear_Probes notebook · **MINE** a decision I made — the audited `safety_probes.ipynb`
design decisions are mine too, so every MINE cites where the plan records the reasoning
(**#N** = numbered revision; *"section name"* = a Pipeline-stages/design-table section of the
plan). "novel" = in no paper and not in [NB].

### What/how we PROBE

| Decision | Choice | Source |
|---|---|---|
| Deception train tokens | fact minus its last 5 **words** (probe sees intent, not the completed lie) | [Apollo §3.2.1], via [NB]/Apollo *code* — words not tokens is the code's behaviour (**#2**) |
| Deception eval tokens | ALL assistant tokens, per-token scores averaged | [Apollo §3.1]; mask util from [NB] |
| Deception: system/user never probed | the persona line IS the label — probing it is leakage | faithful to [Apollo]/[NB]; enforced for the attention probe by fix **#10** |
| High-stakes tokens | content tokens of EVERY role (scaffold/BOS/date excluded), train = eval | **MINE** (**#31**) — disclosed deviation from [HS §2.1] whole-sequence |
| Pooling recipe | per-token training rows, scores averaged at eval ("score-then-average"), both behaviours; `N_TOKEN_CAP=60k` | [Apollo §3.1] recipe; applying it to both behaviours = **MINE** (plan *"Mean pooling: which version, why it's sound"*, decision 2026-07-07) |
| Probe family: MM | plain θ_mm = μ⁺−μ⁻, never Σ⁻¹θ_mm | [GoT §5.1]; App. F for why not Σ⁻¹ (plan *"What mass-mean means here"*) |
| Probe family: LR | logistic, C=0.1, standardized | [Apollo §3.1] via [NB] |
| Probe family: LR-attention | learned attention pooling, 3 seeds ± CI | [HS §2.1] via [NB §5] |
| The 5-probe grid (mean/last × MM/LR + attention) | fully crossed pooling×estimator | **MINE** (plan *"4. Probe grid — 5 probes at the detection layer"*) |
| Detection layer criterion | pair-aware 5-fold CV AUROC of MM, all 32 layers; LR overlaid as check | selection-by-validation follows [Apollo]/[HS]; pair-aware CV = **MINE** (plan *"1. Layer selection"*; listed under "Ours" in the plan's Provenance section); overlay **#16a** |
| Final layer choice | argmax is a *suggestion*; I pick in the EDIT-HERE cell | **MINE** (plan *"You choose the layers from the plots"*; audited: 12 / 20) |
| Splits | 80/20 pair-aware; pair_id groups never straddle folds | **MINE** (plan *"Splits"* note + **#7**) — [NB] used random balanced splits |
| Metrics | AUROC (bootstrap CI) + TPR@1%FPR; accuracy dropped | papers lead with these ([Apollo §3.3], [HS Fig. 3b]); dropping accuracy = **MINE** (**#28**) |
| Deception OOD | AI Liar | [Apollo App. A.1] via [NB §4] |
| Deception OOD #2 | Roleplaying + pool column | **MINE** (**#35**; pool caveats **#49**) |
| High-stakes OOD | Anthropic-HH only; ToolACE removed | [HS Table 2] (partial); removal = **MINE** (**#34**) |
| No control set / no random direction | coherence + 3-direction comparison carry specificity | **MINE** (plan *"Control set"* rows + *"Steering coherence guard"*, decisions 2026-07-07) |
| ID deception framing | sanity check, never a detection claim | [Apollo]'s own train-only treatment; the framing rule = **MINE** (**#20**) |

### What/how we STEER

| Decision | Choice | Source |
|---|---|---|
| Direction arms | MM, LR, LR-attention (CAA arm dropped) | estimators from [GoT]/[Apollo]/[HS]; the fully-crossed arm choice = **MINE** (**#26–#27**) |
| Direction data | refit per block layer from train-split mean-pooled masked activations | per-layer refit = **MINE, novel** (plan *"Steered layers"* row + **#13**; supported by [vdW]) |
| Calibration | coef +1 ≡ cross one class-mean gap, per layer | [GoT §6.1] (plan *"Scale calibration"* row) |
| Application | vector ADDED at EVERY token position, all block layers, all methods — one convention | [CAA]'s add-at-positions convention, generalized; one-convention-for-fairness = **MINE** (plan *"Steered tokens"* row; difference from CAA-main (`[/INST]`+) and [GoT] (statement-end) disclosed, **#33**) |
| Addition only (no ablation) | signed coefficient covers suppression | all four reference implementations add ([CAA], [GoT §6], [vdW], [NB]); removing the dead ablate branch = **MINE** (**#12**) |
| Block criterion | per-layer NIE sweep (MM direction, coef +1), contiguous top window | [GoT §3] "group (b)" analogue via [NB §3]; train-split scope + MM-anchor disclosure = **MINE** (**#16b–c**) |
| Final block choice | `pick_block` is a *suggestion*; I pick in the EDIT-HERE cell | **MINE** (plan *"You choose the layers from the plots"*; audited: [15–19] / [9–13]) |
| Coefficients | {−2 … +2}, signed, 0 = baseline | **MINE** (plan *"Coef sweep"* row — dose-response in both directions) |
| ID-first positive control | steer ID before OOD so a null OOD is interpretable | **MINE, novel** (plan *"Why steer in-distribution first"*) — [GoT] reports OOD only |
| High-stakes steering at all | | **MINE, novel** (plan §5 intro: "no paper steers for these behaviours… a novel contribution") |
| Eval instrument 1: NIE readout | few-shot YES/NO next-token P-diff; gap gate before any NIE | [GoT §6.1] adapted to behaviours; few-shot lock = **MINE** (plan *"Eval (i): NIE"* row, locked 2026-07-07) |
| Eval instrument 2: LLM judge on steered generations | behaviour AND coherence axes | judge use ≈ [CAA]'s GPT-4 eval; the coherence guard = **MINE, novel** (plan *"Steering coherence guard"* + **#30**); CAA-style rubrics **#47**, n=24 **#46**, fail-loud **#38** |
| Deception ID: NIE-only (judge dropped) | prompts byte-identical → judged n≡1 | **MINE** (**#46**; the underlying limitation recorded at **#40b**) |
| Verdict point | NIE at coef +1 (the calibrated unit; +2 is in the coherence-collapse regime) | unit from [GoT §6.1]; pinning the verdict there = **MINE** (**#45–#46**, and `claude-conversations/summary.md` §5) |
| Simultaneous steering | pilot only, gated, not validated | [vdW Fig. 7] analogue; pilot status = **MINE** (**#48**; `compose_steering`'s research-grade flag, plan §4.4) |

## Method core (the plan's design-table decisions)

Decisions recorded in the plan's *Model & infrastructure* / *Pipeline stages* sections — mostly
made before the conversation record begins, so their provenance is the plan itself.

- **Model: `Llama-3.1-8B-Instruct`, 8-bit, loaded once.** What [NB] used and what was cached on
  the box; [HS §B.7] gives an 8B reference point — expect (and disclose) weaker numbers than the
  papers' 70B. → plan *Model & infrastructure*.
- **Judge: Anthropic API (Claude), locked 2026-06-30**; cells skip cleanly without a key, NIE
  still fills. Model now `claude-haiku-4-5` (`judge.py`).
- **Disk activation cache, one entry = all layers pooled** — re-running on Vast must not re-run
  the model, and it makes the SAE extension pure post-processing. Key is deliberately simple and
  **content-blind** (the documented trap, **#5**).
- **Uncertainty: bootstrap 95% CIs on AUROC; attention probe = 3 seeds ± CI** — [HS Fig. 2]'s
  convention; AI Liar is too small for defensible point estimates. Per-seed AUROCs printed so
  seed instability can't hide inside the ensemble (**#18a**).
- **Mean pooling = the Apollo recipe (score-then-average), applied to BOTH behaviours.** Two
  versions exist in the literature: (a) [HS]-style average-then-score (one row per example) and
  (b) [Apollo §3.1]-style per-token training (one row per token; eval averages the scores). For a
  fixed linear probe they are *identical at evaluation* (`mean(w·xᵢ) = w·mean(xᵢ)`), so (b) was
  chosen: one recipe everywhere, exact Apollo fidelity for the probe being replicated, and it's
  what makes per-token heatmaps meaningful. `N_TOKEN_CAP = 60000` caps the token-row table
  (~15 rows/weight, uniform over tokens, training-only). → plan "Mean pooling: which version,
  why it's sound" (decision 2026-07-07, supersedes "pooled only").
- **Plain θ_mm everywhere; never Σ⁻¹θ_mm.** [GoT App. F]: the Σ⁻¹-corrected variant converges to
  the LR direction — it would duplicate the LR arm — and plain θ_mm is the OOD/causal-appropriate
  one. → plan "What mass-mean means here".
- **Few-shot (not zero-shot) readout, locked 2026-07-07** — readout reliability at 8B; the
  YES/NO next-token format is [GoT §6.1]'s calibrated-readout analogue for behaviours the model
  never emits as a token.
- **Coef sweep {−2, −1, −0.5, 0, +0.5, +1, +2}** — dose-response in both directions (induce and
  suppress); 0 is the baseline point, measured once and shared (**#19**).
- **Secondary concatenated-features comparison: DROPPED** (**#51**, my decision 2026-07-30).
  Locked 2026-06-30 but never implemented; no result ever depended on it. The per-layer steering
  block already carries the multi-layer story.

## Data & masking

- **Use only the 306 true facts; the `[:512]` cap is a no-op.** I challenged the claimed 512
  ("can you confirm there are 512 true facts…", `b06abdf5` L634) — the CSV has 306 true rows.
  → **#1**. All counts in the plan corrected.
- **Drop the last 5 *words* (not tokens) of the fact at training.** The paper's prose says
  "tokens", but both reference implementations (Apollo's `_split_statement`, NB) drop words —
  match the code that produced the published numbers, footnote the prose discrepancy.
  Asked at `7018fd8b` L649 ("tokens vs words? It's unclear to me") → **#2**.
- **Deception probes read assistant tokens only — never the persona line.** The system prompt IS
  the label; probing it is leakage. My question "why do we not probe 'you are a deceptive
  agent'?" (`7018fd8b` L8218) confirmed the policy; fix **#10** had earlier stopped the attention
  probe from seeing it.
- **High-stakes mask: whole-sequence → content tokens of every role.** I pushed on the template
  tags ("Any way we could mask it… How does this affect ood?", `7018fd8b` L7699); measurement
  showed scaffold was 53%/37%/7% of tokens across sets — a dataset-dependent dilution baked into
  the OOD comparison. Disclosed deviation from [HS §2.1] → **#31**.
- **Why deception and high-stakes mask differently at all** (my question `7018fd8b` L9410 item 3):
  the behaviours' ontologies differ — deception lives in the reply given its context ([Apollo]
  scores assistant tokens); stakes live in the situation ([HS] scores the interaction). Recorded
  as rule 1 of data_reference's "The four rules".
- **`data_reference.md` exists because I demanded it** ("i'M STILL not sure what the data looks
  like…", `7018fd8b` L7978; then twice: "use actual examples" L8136) → **#32**. Keep it updated
  when masks or loaders change.
- **Data vendored and pinned** (Apollo files at commit `f8ec4010` + SHA-256; MUP by revision).
  A thesis needs frozen inputs; the upstream clone was 535 MB and unpinned → **M8**.
- **ToolACE removed from high-stakes OOD** — tool-calling is a different interaction modality,
  out of scope; comparability with [HS Table 2] narrows to its Anthropic-HH column → **#34**.
- **Roleplaying added as second deception OOD** (detection **#35**, steering **#39**) — resolves
  the degenerate TPR@1%FPR on 27-negative AI Liar and buys the context-defined vs content-defined
  near/far generalization contrast.
- **Sample sizes 1500/500 per class for high-stakes** — matching NB §5, replacing an undocumented
  smaller deviation → **#7**. Splits pair-aware everywhere (pair_id groups never straddle folds).

## Detection & metrics

- **One detection layer, chosen by pair-aware 5-fold CV AUROC (MM as selector, LR overlaid).**
  I asked whether single-layer detection was well-founded vs aggregating (`37b6179a` L612):
  reading is passive — one snapshot suffices if the layer is validated; and one layer gives one
  direction to steer/decompose. Layer choice is a visible EDIT-HERE constant, never silent.
  Selection hardening (train-only sweeps, LR overlay, MM-anchor disclosure) → **#16**.
- **AUROC (bootstrap CI) + TPR@1%FPR; accuracy dropped.** Frozen-threshold accuracy measures
  threshold *transfer*, not representation quality (the 0.992-AUROC/0.500-accuracy artifact).
  I initially wanted AUROC-only (`7018fd8b` L7518), revised to keep TPR@1%FPR for number-level
  comparability with [HS Fig. 3b] → **#28**. TPR returns NaN under 100 negatives — honesty over a
  degenerate number (**#35**).
- **No control set / no random-direction arm** (locked 2026-07-07): the coherence axis plus the
  three-direction comparison carry the specificity burden. (A transfer/selectivity matrix remains
  the ready answer if an examiner pushes — discussed 2026-07-30, deliberately left out.)
- **ID deception numbers are a sanity check, never a detection claim** — the persona feature makes
  ceiling AUROC the *expected* result; framing rule → **#20**.
- **Probe-direction cosine matrix added; cross-behaviour cosines rejected** — within one layer the
  5×5 cosine matrix is the mechanism check for the read/write hypothesis; across behaviours the
  detection layers differ, so vector comparisons would cross residual bases → **#41**.
  Reaffirmed 2026-07-30 (probe similarities left out of the comparison notebook, **#48**).

## Steering design

- **CAA arm dropped; LR-attention arm added — fully-crossed read/write matrix.** The outcome of my
  hardest-won understanding (five turns, `7018fd8b` L7041–L7367; decision at L7388): CAA is the
  same difference-of-means estimator at a different extraction site, so the site comparison was
  non-central; with LR-attention steered, every detector has a steering counterpart → **#26–#27**.
- **Extraction varies by method; application never varies.** Every direction is added at every
  token position of the block — one convention so the comparison is fair at intervention time.
  Rooted in my "we steer on every token right?… CAA-main steers a smaller subset?" (`7018fd8b`
  L8218 item 3): CAA-main steers from `[/INST]` on, GoT only statement-end — ours deliberately
  differs, disclosed and visualized → **#33**.
- **Addition only; `steer_mode`/ablation removed.** My call ("do you think we can do away with
  steer_mode?", `7018fd8b` L2716): all four reference implementations steer by signed addition;
  the ablate branch was untested dead code in the class that mutates the residual stream.
  Caveat recorded for a future refusal behaviour (ablation ≠ negative addition) → **#12**.
- **Per-layer vectors, per-layer calibration** (vs GoT's single vector at all layers): the
  residual basis drifts and norms grow with depth — layer-local calibration keeps "coef 1.0 =
  one class gap" literally true at every steered layer; independent support in van der Weij →
  plan "Steered layers" row + **#13**.
- **Mean-pooled direction variants only; last-token variants not steered** — measurement/write
  matching: we write to every token, so measure the arrow over typical tokens → **#26**.
- **Contiguous 5-layer block** (my question "any reason the 5 layers have to be contiguous?",
  `7018fd8b` L6685): contiguity is a simplicity/comparability choice around the NIE peak, not a
  law; `pick_block` suggests, the EDIT-HERE cell decides.
- **ID-first steering as positive control** (our methodological addition): a null OOD is ambiguous
  between "not causal" and "doesn't transfer"; ID answers "does the knob turn at all". OOD stays
  the headline, following [GoT].
- **Verdict pinned at coef +1**: the calibrated natural unit (+1 = one class gap); +2 lives in the
  coherence-collapse regime where verdicts reward breaking the model. (My question `c8aa4b41`
  L28401, answered 2026-07-30 — `claude-conversations/summary.md` §5.)

## Evaluation instruments

- **Two instruments, strict division of labor**: the NIE readout (Llama's own YES/NO next-token
  judgment; concept-level, causal, powers all sweeps) vs the Claude judge (scores steered
  *generations*; behaviour-level corroboration). The judge never feeds layer selection or the
  verdict — data_reference "The four rules", rule 4.
- **Judge must not fail quietly** — my requirement verbatim ("I don't want the judge to break
  quietly", `7018fd8b` L10735): shortfall warnings + 0/N raises → **#38**. Proved necessary by
  **#44** (the schema bug that had silently 400'd every call — found because failures were loud
  in stdout).
- **Judge n: 6 → 24** — CAA averages 50 judged answers per point; 1–10 integer scores at n=6
  cannot make smooth curves. Deception ID judging dropped entirely (prompts byte-identical →
  effective n=1 at any budget); its NIE positive control stays → **#46**.
- **CAA-style rubrics** (~100 words, both ends anchored, intermediate guidance — verbatim
  comparison in `sketchbook.md`); output format kept ours (enum-bounded JSON, coherence axis).
  Judge validation moved to neutral-instruction OOD prompts so the persona line reaches no judge
  input anywhere → **#47**.
- **Deception ID generation keeps the deceptive instruction** — it is constant across
  coefficients, so the *relative* dose-response survives; read that curve for shape and
  suppression only. My "won't the judge know it's deceptive?" (`7018fd8b` L6804 →
  "I think I don't need to do anything, things get normalized" L11959) → **#40**.
- **Coherence panel is mandatory** — collected-but-unreported coherence violated the plan's own
  spec; now the third panel everywhere → **#30**. CAA has no coherence axis (they curated the
  multiplier range by hand); ours quantifies the same guard.

## Reporting & structure

- **`summary.ipynb` dissolved** — verdicts belong at the end of each behaviour's probing/steering
  sections; win-shares are now per-behaviour (cleaner for "does it hold across behaviours");
  found+fixed: Roleplaying had been silently missing from the saved steering verdict → **#45**.
- **`comparison.ipynb`**: MM direction only, OOD sets only, NIE in class-gap units as the
  cross-behaviour-comparable axis; simultaneous-steering pilot gated and labelled; specs moved to
  `behaviours.py` as single source of truth → **#48**.
- **OOD pool: concatenation kept for now, macro-average deferred post-run** — the pool is 93%
  Roleplaying and pooled AUROC mixes in cross-set score calibration; per-set columns are the
  load-bearing, paper-comparable results (neither paper pools) → **#49**.
- **Names I insisted on**: `steering_nie_llm_evaluation` (both measurements in the name; my
  naming, `7018fd8b` L4144+), `extract_full_sequence_one_layer` (true counterpart of
  `extract_pooled_all_layers`; **#11**), repo `probing-safety-behaviours` (distinct from the
  package name; **M14**).

## Presentation & figures (plan-recorded)

- **Notebook restructured behaviour-major, not stage-major** (**#14**: 50 → 89 cells; §1 shared
  helpers, then §2/§3 as true mirrors) — "a new behaviour = append a section". The 2026-07-30
  heading harmonisation of the driver notebooks continues this.
- **Heatmaps follow [NB solutions]' 3-phase pattern** (score → shared centering from masked
  tokens → render), masked to detection-mask ∩ non-special tokens — fixing BOS-sink saturation
  and missing centering (**#23**); attention-weight heatmaps added as the "where does the probe
  look" panel (**#25**), extended to OOD (**#36**); `show_qualitative` broadened to all three
  directions at coef +1 (**#37**); ROC curves added as the visual companion of the AUROC table
  (**#29**).
- **Matplotlib/GitHub-renderable figures** (**M15**): Plotly/circuitsvis don't render on GitHub;
  converted with the seaborn *look* without a seaborn dependency, dual circuitsvis+PNG token
  heatmaps sharing one transform; retina/DPI pairing completed **#50** (solid ROC lines — the
  dashed variant was tried and rejected).
- **Judge-validation protocol is blind by design** (**#18b**): label the CSV before looking at
  the judge's scores; agreement = Spearman ρ / mean|diff| / %±2.

## Engineering & reproducibility

- **All tokenization through `chat.py`** — one token of template/mask disagreement corrupts every
  probe score silently; the double-BOS bug (**#22**) is why. Verified to survive the extraction
  (**M2**); I demanded the re-verification ("is the double bos quirk still fixed though?",
  `c8aa4b41` L18154).
- **Content-blind activation cache** — deliberate simplicity with a documented trap: edit a
  loader → delete the matching `activation_cache/*.pt` (**#5**; cache wiped at **#22** and again
  at the #31 mask change).
- **Export calibrated directions, never raw** — calibration is baked into the vector; exporting
  raw would silently break every downstream strength control → **#43**. Bundles refuse a
  model/d_model mismatch (insurance the cache lacks).
- **Parity gate** — the package must reproduce the audited notebook before replacing it; on the
  new instance the bar is *same choices + CIs overlap* (torch/GPU changed), with a torch-2.11 pin
  as the exact-parity fallback → plan Status 2026-07-30.
- **Seed 42 via `runtime.init()`; every stochastic step also uses its own explicitly-seeded
  generator** — results independent of cell execution order.
- **Notebook globals → `runtime` module, not threaded arguments** (**M1**): threading
  model/tokenizer through every signature would have made the audit diff unreadable; cost =
  `runtime.init()` must run first (accessors fail loudly if not).
- **Loaders return `Split` objects — the one deliberate API change** (**M6**): a module cannot
  leave ~30 globals behind as an import side effect; everything else about
  tokenization/splitting/sampling is unchanged, which is why the parity gate is a full re-run
  rather than a diff.
- **FastAPI, not Gradio, for the UI** (**M9**): gradio 6 requires `huggingface_hub>=1.2` vs
  transformers needing `<1.0` (would have broken model loading); gradio 5 downgrades six
  packages. FastAPI/uvicorn were already installed — zero new dependencies.
- **OOD full-sequence tensors are RAM-only** (`use_cache=False`, **#24**): ~8 GB per config that
  would overflow the disk; ~10 min to recompute on kernel loss is the accepted trade.
- **`.env` is repo-local only** (**M16**, extended 2026-07-30 with `HF_HOME`): the package never
  walks parent directories, so the repo is standalone; Jupyter kernels don't inherit Vast's env,
  so the repo `.env` is the reliable channel.
- **Advisor protocol** — "I don't want you to come up with a plan on your own… You're my advisor"
  (`7e01cf60` L875); don't fix what wasn't asked (`7018fd8b` L1001); verify against primary
  sources before answering (standing instruction, 2026-07-30). These are why this file can cite
  its provenance.
