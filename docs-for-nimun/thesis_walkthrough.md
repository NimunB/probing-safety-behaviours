# Probing & Steering Safety Behaviours — a walkthrough

A guided tour of this thesis for a new reader: the research question, the concepts, the technical
machinery, and the choices — with pointers into `safety_probes_plan.md` (decision numbers like
**#31**), `data_reference.md` (prompt-level ground truth), and the code (`safety_probes/`).
Written 2026-07-30, before the first full run on the new instance.

---

## 1. The question

Two findings from the literature sit in tension:

- **Detecting** deception: logistic-regression probes are the best *readers* of model internals
  ([Apollo, App. D.1] — Goldowsky-Dill et al., *Detecting Strategic Deception Using Linear Probes*).
- **Steering** truth: difference-of-means directions are the most *causal* — they move behaviour
  best when added to activations ([GoT, Table 2] — Marks & Tegmark, *The Geometry of Truth*).

Same activations, similar classification accuracy — yet the best direction for *reading* a concept
may not be the best handle for *writing* it. **Does this read/write dissociation hold across
safety-relevant behaviours with different ontologies?** We test it on two behaviours (a third,
sycophancy, is planned):

| | Deception | High-stakes | Sycophancy (built 2026-07-31, first run pending) |
|---|---|---|---|
| Positive class | dishonest reply (1) | high-stakes situation (1) | reply echoing a wrong user claim (1) |
| Where the behaviour "lives" | in the reply, relative to its instruction | in the situation being described | in the reply, relative to the user's claim |
| Encoded in | the **instruction** (persona line) | the **prompt content** | the **interaction** (claim × reply) |
| Anchor | [Apollo] | [HS] (McKenzie et al.) | disentangle-sycophancy data (arXiv 2509.21305); design ours — `sycophancy_plan.md` |

Model: `meta-llama/Llama-3.1-8B-Instruct`, 8-bit, loaded once (`runtime.init()`). Everything runs
on one 24–32 GB GPU.

## 2. The objects

**Activations.** For every prompt we run one forward pass and read the residual stream — a
`d_model=4096` vector per token per layer (32 transformer layers; the embedding output is
excluded, `extraction.py:90`). Probes and steering both
operate on these vectors. Extraction: `extraction.py` (`get_pooled` = mean & last-token vectors at
*all* layers per example; `extract_full_sequence_one_layer` = every token at *one* layer, for the
attention probe and per-token analyses). Cached to disk; **the cache key is content-blind** —
edit a loader, delete the matching `activation_cache/*.pt` (plan "Cache caveat").

**Masks.** Not every token should be read. Each behaviour has a mask policy (`chat.py`):
deception = `assistant_only` (the reply), high-stakes = `content_tokens` (**#31**: every role's
content, minus BOS/template scaffold — scaffold was up to 53% of tokens and diluted the OOD
comparison). All tokenization goes through `chat.py`'s helpers; calling `apply_chat_template`
directly reintroduces a fixed double-BOS bug (**#22**). Prompt-level ground truth with real
highlighted examples: `data_reference.md`.

**Probes (the READ side).** Five per behaviour, all at one chosen "detection layer"
(`probes.py`, `metrics.build_probe_grid`):

| Probe | Direction from | Pooling |
|---|---|---|
| MM (mean / last) | closed-form θ = μ⁺ − μ⁻ (difference of class means) | mean over masked tokens / last masked token |
| LR (mean / last) | logistic regression (C=0.1, [Apollo §3.1]); LR (mean) is trained **per-token** — every masked token is a training row (Apollo's recipe; also what makes token heatmaps meaningful) | mean / last |
| LR-attention | learned: `W_q` scores each token's relevance, softmax → weights, weighted sum → `W_out` classifies ([HS §2.1]); 3 seeds ± CI | learned attention |

Why paired data matters: differencing class means over *pairs* (same fact, honest vs deceptive
persona; same situation, high vs low stakes) cancels shared content and isolates the behaviour
axis. A probe yields a **direction, not a verdict** — scores are uncalibrated projections;
we report threshold-free metrics.

**Steering vectors (the WRITE side).** The same three direction families (MM, LR, LR-attention's
`W_out`), refit **per layer** of a chosen 5-layer block (`steering.build_directions`), each
**calibrated** so that coefficient +1 moves activations exactly one class-mean gap
([GoT §6.1]; `steering.calibrate`). Applied by a forward hook that **adds the vector at every
token position** of every block layer (`SteeringHook`) — one fixed application convention so the
three directions compete fairly (extraction site varies by method; application never does).
Signed coefficients −2…+2 test induction *and* suppression.

## 3. The data and the OOD design

- **Deception train/ID**: 306 true facts × {honest, deceptive} system persona = 612 paired
  examples (Zou et al. format, [Apollo §3.2.1]); train masks the fact minus its last 5 *words*
  (**#2** — matching Apollo's code, not its prose), eval scores all assistant tokens.
- **Deception OOD**: *AI Liar* (27 scenarios; **context-defined** — byte-identical answers, only
  the instruction flips, so the probe must read intent from context) and *Roleplaying* (371;
  **content-defined** — the lie is in the completion text) (**#35**). Near/far generalization
  gradient; the pooled column is 93% Roleplaying, so per-set numbers are load-bearing (**#49**).
- **High-stakes train/ID**: `Arrrlex/models-under-pressure` synthetic split, grouped by `pair_id`
  (all tone/language variants of one situation pair stay together — splitting them is leakage).
- **High-stakes OOD**: `anthropic_hh_balanced` — real human conversations (ToolACE removed, **#34**).
- All splits **pair-aware** 80/20; data vendored and pinned (Apollo commit + `MUP_REVISION`, **M8**),
  SHA-256 in `data/PROVENANCE.md`. Seed 42 everywhere (`runtime.py`), every stochastic step
  explicitly seeded.

## 3b. Why exactly these tokens — the masking rationale, with divergences from the sources

The principle behind every mask: **probe the tokens where the behaviour lives, and nothing that
gives the label away.** Steering deliberately ignores masks — one write-convention for all
methods. Per behaviour:

### Deception — probes read the assistant's reply; training stops 5 words early

- **Training: the fact minus its last 5 words.** Why: the probe should read *intent to deceive*,
  not a completed lie — truncated, the statement could still be finished truthfully or
  deceptively, so what separates the classes at those positions is only the persona context the
  model is carrying. Source: [Apollo §3.2.1]. **Divergence, disclosed:** the paper's *text* says
  drop 5 "tokens"; both reference *implementations* (Apollo's `_split_statement`, [NB]) drop 5
  space-separated **words** — we match the code that produced the published numbers and footnote
  the prose discrepancy (**#2**).
- **Never the system/user turns.** The persona line ("You are a deceptive AI assistant") *is* the
  label — probing it is leakage, not detection. This was enforced the hard way: fix **#10** caught
  the attention probe reading the persona and inflating its numbers. Faithful to [Apollo]/[NB].
- **Evaluation (ID and OOD): ALL assistant tokens, per-token scores averaged.** Why the
  asymmetry: at deployment you don't know where in a response the lie is, so you score everything
  the assistant said — [Apollo §3.1]'s exact recipe. The train-style truncation exists only where
  intent must be isolated from content. Consequence for framing: ID numbers are a sanity check,
  never a detection claim (**#20**).
- **OOD keeps the same all-assistant policy** — which is what makes AI Liar informative: its
  answers are byte-identical across honest/deceptive framings, so any score difference must come
  from the probe reading *context through attention*, not answer wording.

### High-stakes — probes read every role's content; train = eval

- **Content tokens of ALL roles.** Why: stakes are a property of the *situation described*,
  overwhelmingly in the user's message — an assistant-only mask would read the reaction, not the
  situation. There is no train/eval asymmetry because there is no intent-vs-execution distinction
  to preserve: the situation is fully stated either way.
- **Divergence from [HS §2.1], disclosed (#31):** the paper pools the **whole sequence**,
  template scaffold included. We exclude BOS, header tags, and the auto-injected date preamble,
  because scaffold was measured at **53% / 37% / 7%** of tokens across our eval sets — a
  stakes-irrelevant, *dataset-dependent* dilution that whole-sequence pooling bakes into any
  ID-vs-OOD comparison — and BOS is a known attention-sink outlier. Multi-turn OOD conversations
  keep every role's content (matching [HS], which computes over the real dialogue — itself a
  fidelity fix over [NB], which fed the raw JSON string as one user turn).

### Sycophancy — probes read the assistant's reply only; train = eval; no truncation

- **Assistant reply content only.** Why: the behaviour is *relational* — the same reply is
  sycophantic or candid depending on the user's claim — so the reply is where it must be read,
  with the claim reaching the probe only **through causal attention**, exactly the mechanism
  AI Liar tests for deception, here used at training time. Masking in the user turn would hand
  the probe a shortcut (claim-correctness correlates with the label).
- **No drop-last-k.** Apollo's truncation exists because an unfinished statement can still go
  either way; sycophancy replies are complete and the label comes from context, so the
  symmetric-mask policy (like high-stakes) is the faithful choice. `sycophancy_plan.md` §5c.
- **Same mask ID and OOD (math)** — the OOD claim is domain generalization, so nothing else may
  vary. The genuine-agreement hard negatives share their reply text with sycophantic rows almost
  exactly, which is the designed proof that the mask + attention combination, not reply wording,
  carries the signal (the §5.i anti-echo-shortcut check quantifies this).

### Steering — every token, always, for every behaviour

- The steering vector is added at **every position** (prompt + generated, few-shot readout prefix
  included), across the block, for all three directions. Why: the three directions are *extracted*
  at different sites (mean over masked tokens; LR-attention's pooled context), and the comparison
  is only fair if they are *applied* identically — "extraction varies by method; application never
  varies."
- **Divergences, disclosed (#33, verified against both codebases):** CAA-main steers only from
  `[/INST]` onward; [GoT] steers only statement-end tokens. Ours deliberately differs for
  fairness across methods; visualized in `data_reference.md` (fully-highlighted blocks = steered)
  and worth one disclosure sentence in the thesis (the readout's few-shot prefix is steered too —
  constant across items and coefficients, so it cannot bias the dose-response).
- Why classification and steering token sets differ *on purpose*: we **read** a summary from
  pooled tokens, but **write** to the whole stream so the intervention shapes generation — plan
  "Exactly which tokens" section.

Where this is recorded in the decision record: plan *"Exactly which tokens are used"* schematics,
the behaviour tables' *Probed tokens* rows, revisions **#2, #10, #20, #31, #33**, and
`sycophancy_plan.md` §5c; per-decision provenance in `decisions.md`.

## 4. The pipeline, stage by stage

Each behaviour notebook (`notebooks/deception.ipynb`, `high_stakes.ipynb`) runs the identical
skeleton (§1–§8):

1. **Spec + data** — `BehaviourSpec` (in `behaviours.py`, **#48**) configures masks, the readout,
   the judge rubric, coefficients.
2. **Extraction** — pooled activations, all layers, cached.
3. **Layer selection (READ)** — pair-aware 5-fold CV AUROC of an MM probe per layer
   (`metrics.cv_auroc_by_layer`), LR overlaid as a robustness check; argmax is a *suggestion*,
   the choice is a visible EDIT-HERE constant (audited: 12 / 20).
4. **PCA** — caveat built in: PCA shows variance, which can ride topic confounds
   ([Apollo §4.1.2]); the evidence a linear probe exists is the projection histogram onto the
   trained direction. (For deception, PCA showing *no* separation is the good outcome — the
   signal is real but low-variance.)
5. **Probing** — the 5-probe grid; evaluation on ID test + OOD sets: **AUROC (bootstrap 95% CI)**
   + **TPR@1%FPR** ([HS Fig. 3b] comparability; NaN under 100 negatives — honesty over
   degeneracy, #35). Accuracy deliberately dropped (**#28**: threshold transfer ≠ representation
   quality). Probing verdict cell (**#45**): best reader per eval set + LR-family win share.
6. **Heatmaps** — per-token LR scores (where the probe fires) and attention weights (where the
   best detector looks). Colors are relative shading, not verdicts.
7. **Steering (WRITE)** —
   - **7a. Block selection**: per-layer NIE sweep with the calibrated MM direction on
     *train-split* transcripts (selection never touches eval items, **#16b**); contiguous top
     window suggested, EDIT-HERE (audited: [15–19] / [9–13]). MM-anchoring disclosed.
   - **7b. Directions**: per-layer refit + calibration for all three methods.
   - **7c. Evaluation, two instruments** (the load-bearing distinction):
     * **NIE readout** — Llama's own next-token judgment. Build few-shot + transcript +
       "Was the assistant deceptive? YES or NO", read P(" YES")−P(" NO") at the last position
       with steering active (`steering.readout_pdiff`). NIE = shift from baseline, normalized by
       the unsteered class gap — which is a *gate*: near-zero gap → NIE undefined, say so
       (`nie_baseline_gap`). **No generation. Measures whether the direction shifts the model's
       perception of a fixed text — concept-level causality.**
     * **LLM judge** — Claude Haiku scores *steered generations* on two axes: behaviour and
       **coherence** (1–10, CAA-style rubrics **#47**, enum-bounded JSON **#44**, n=24 per point
       **#46**). **Measures whether steering changes what the model writes — behaviour-level.**
       Coherence is the guard: a behaviour "win" at collapsed coherence is steering-by-breaking.
     * ID runs first as **positive control** (our methodological addition — makes a null OOD
       interpretable), then OOD carries the claim. Deception ID is NIE-only (**#46**: its
       generation prompts are byte-identical → judged n would be 1 at any budget).
   - **7d. Steering verdict** (**#45**): best writer per eval set by NIE at coef **+1** (the
     calibrated unit — +2 lives in the coherence-collapse regime) + MM win share + the
     per-behaviour dissociation line.
8. **Export** — probes, calibrated directions, layer/block choices → `exported_probes/*.pt`
   (~0.5 MB; the only GPU-expensive committed artifact) + CSVs/JSON of final numbers.

Downstream: `comparison.ipynb` (**#48**) overlays behaviours' dose-response curves (CAA Fig. 5
analogue) and drafts simultaneous steering (van der Weij Fig. 7 analogue; blocks are disjoint);
`ui_probe.py` serves live detection/steering from the exported bundles; `judge_validation.ipynb`
measures human–judge agreement blind (run once before quoting judge numbers).

## 5. The choices ledger (what an examiner will ask about)

Compact version — the full per-decision record with rationale and provenance (plan entry numbers
**and** the transcript locations where each call was made) is `decisions.md`.

| Choice | Decision | Why (plan ref) |
|---|---|---|
| Layer for reading vs layers for writing | 1 CV-chosen layer to detect; 5-layer NIE-chosen block to steer | reading is passive, one snapshot; writing is causal and attenuates — different criteria for different jobs (plan "Layer selection") |
| Plain θ_mm, never Σ⁻¹θ_mm | plain everywhere | Σ⁻¹θ_mm converges to LR ([GoT App. F]) — would duplicate the LR arm |
| Per-token LR (mean) training | Apollo's exact recipe | eval is identical either way (linearity); enables token heatmaps (plan "Mean pooling") |
| CAA arm dropped | MM/LR/LR-attention, fully crossed | CAA = same estimator, different site; crossing read & write directions answers the thesis question (#26–27) |
| Steer every token, per-layer vectors | fixed convention; layer-local direction + calibration | fairness across methods; "coef 1 = one gap" true at every layer (#12–13, plan "Steered layers") |
| Addition only, no ablation | signed addition | all four reference implementations steer by addition; ablation reserved for refusal (#12) |
| No control set / no random-direction arm | coherence axis + 3-direction comparison carry specificity | locked 2026-07-07; transfer-matrix idea noted as future hedge |
| AUROC + TPR@1%FPR, no accuracy | threshold-free metrics lead | #28; TPR@1%FPR kept solely for [HS] comparability |
| Judge: n=24, both-ends rubrics, coherence axis, fail-loud | | #38, #44, #46, #47 — CAA comparison in `sketchbook.md` |
| ID steering kept, ID judging dropped (deception only) | NIE positive control stays | #46 |
| Verdicts inline per behaviour; comparison notebook cross-behaviour | | #45, #48 |
| Data vendored + pinned; bundles committed; caches not | | M8, #43 |

## 6. Framing rules and limitations (say these before the examiner does)

- Deception **ID detection numbers are a sanity check, never a detection claim** — the persona is
  the label (#20). The reportable numbers are OOD.
- **NIE ≠ "X% of responses became deceptive."** It is a perception-shift in class-gap units. The
  response-level evidence is the judge-behaviour panel with coherence intact.
- Judge scores are single-source until `judge_validation` runs; rubric-dependent, so compare
  shapes across behaviours, not levels.
- Off-policy OOD (prewritten text, not 8B-generated); AI Liar n=27; 8B vs papers' 70B; deception's
  baseline readout gap is thin (+0.065 in the audited run) — watch it.
- The parity bar on the new instance is *same choices + CIs overlap*, not bit-parity
  (torch 2.12/cu130 + RTX 5090 vs the audited 2.11/cu128 + 3090).

## 7. Repo map

```
safety_probes/            the package (each module docstring names its notebook section)
  runtime.py              model/tokenizer/seed/env; call runtime.init() first
  chat.py                 templating + masks — the load-bearing tokenization layer
  behaviours.py           DECEPTION_SPEC / HIGH_STAKES_SPEC (single source of truth)
  extraction.py           cached activation extraction (content-blind key!)
  probes.py  metrics.py   probe classes; grid, AUROC/TPR, CV layer sweep
  steering.py             hooks, calibration, readout, NIE sweep, direction building
  judge.py  evaluation.py Claude judge; the dose-response evaluation loop
  summary.py  export.py   verdict helpers + artifact persistence; probe bundles
  viz.py                  all figures
notebooks/                deception / high_stakes (identical skeleton §1–§8),
                          comparison (#48), judge_validation
data/                     vendored, pinned datasets (PROVENANCE.md)
exported_probes/          bundles + final-number CSVs/JSON
docs-for-nimun/           the plan (decision record), data_reference (prompt-level truth),
                          papers/, sketchbook (CAA judge comparison), this file
ui_probe.py               FastAPI live detection/steering over the bundles
```

**Reading order for a newcomer:** this file → `data_reference.md` (see what the model actually
sees) → plan's "Pipeline stages" → the deception notebook top-to-bottom → the plan's numbered
revisions for any decision you want the history of.
