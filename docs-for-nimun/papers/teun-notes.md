# Notes: van der Weij et al. 2024 — *Extending Activation Steering to Broad Skills and Multiple Behaviours*

`multiple-steering-teun.pdf` (arXiv:2403.05767). Read 2026-07-11 during the `safety_probes.ipynb`
audit; referenced by the plan's Jul 10–11 revisions #12–13. Notes relate the paper to
`safety_probes.ipynb` / `safety_probes_plan.md` ("ours" below).

## What the paper does

Two experiment groups, both on Llama-2-7B-Chat, both **pure signed activation addition**
(coefficient-scaled add/subtract — zero mentions of ablation/projection anywhere).

### 1. Broad steering (skills)

- Question: can you steer *away* a broad skill (general coding) as effectively as a narrow one (Python)?
- Vectors: mean of **last-token** residual activations over 5,000 Pile samples per domain (text / code / Python).
- Intervention: "add text vector, subtract code (or Python) vector" at a single layer; coefficient sweep.
- Metric: relative top-1 next-token accuracy on code vs text — their operationalization of **alignment tax**.
- Findings: broad ≈ narrow (contradicting their own hypothesis). E.g. −10% coding accuracy for only −3% text
  at good layers. **Permuted-vector control** (same mean/std, shuffled coordinates) behaves like bad layers.
  Layer 0 steering produced the *opposite* effect (they speculate copy-suppression; unresolved).

### 2. Multi-steering (behaviours)

- Five Perez et al. (2022) A/B-question behaviours: myopia, wealth-seeking, political sycophancy,
  agreeableness, anti-immigration. 1000 samples each (500 vector-generation / 200 val / rest test).
- Vectors: CAA-style contrastive (matching minus non-matching answer prompt), last token.
- Per-behaviour grid search over injection coefficient and layer; **metric = matching score** (share of
  A/B answers matching the behaviour).
- Hygiene filters: discard hyperparameters if >5% of outputs aren't valid A/B tokens (faulty answers),
  or if one valid token exceeds 95% frequency (**mode collapse** — "a broken model, not a steered one").
- Two ways to steer all five at once:
  - **Combined** (merge 5 vectors into one; 8 schemes: mean/sum × weighted/unweighted × add/subtract):
    **largely fails** — smaller/erratic effect sizes, sign flips (anti-immigration), mode collapse.
  - **Simultaneous** (each behaviour's vector at a *different layer*, 11–15, one global coefficient −2..+2):
    **works** for myopia + wealth-seeking, weaker/unstable for agreeableness + anti-immigration,
    sycophancy null either way. Alignment tax only a couple % at |coef| = 1.
- Conclusion: interaction effects *between layers* are less damaging than corrupting the direction by
  vector-merging. Simultaneous > combined.

## Relation to our work

### Direct support (logged in plan revisions #12–13)

- **Addition-only convention** (revision #12): the paper is CAA-family, signed addition with coefficient
  sweeps — same convention as our `SteeringHook` after `steer_mode` removal. "Subtraction" = negative coef.
- **Per-layer simultaneous injection** (revision #13): best external support for our per-layer steering
  vectors. Their own prior hypothesis was that cross-layer interaction effects (steering layer 11 changes
  the inputs layer 12's vector was made for) would hurt — their result says the compounding is tolerable,
  while *merging* vectors is not. **Caveat for the thesis**: they inject five *different behaviours* at five
  layers; we inject *one behaviour's* layer-local sibling vectors across a contiguous block. Supporting
  evidence, not a replication.

### Methodological rhymes (citable)

- Their **matching score on constrained A/B answers** = same readout philosophy as our few-shot YES/NO
  NIE readout (forced-choice token instead of judging free text). Our planned sycophancy extension
  (A/B readout targets) is literally their format.
- Their **faulty-answer + mode-collapse filters** = cheap cousin of our judge's coherence axis. Worth
  stealing as a nearly-free readout sanity check: if steered P(YES)−P(NO) saturates identically across
  all items, that's mode collapse, not behaviour shift. (Suggestion only — not implemented.)
- Their **alignment tax** (top-1 accuracy on neutral Pile text per coefficient) is a stricter
  general-capability control than judge coherence. We decided the coherence axis carries this burden;
  if a reviewer pushes, "add a Pile top-1 curve per coefficient" is the ready answer and this paper is
  the method citation.

### Warnings

- Their **sycophancy steering failed** (dataset was political-topology-specific) — behaviour steering
  lives or dies on dataset construction.
- Their grid-searched coefficients vary wildly (0.5 to 20+) because they **don't normalize vectors**;
  their stated future work is per-vector coefficients. Our **mean-gap calibration is exactly that
  improvement** — the thesis line for why our coef 1.0 means the same thing at every layer and for
  every direction, where theirs doesn't.

### Weight of evidence

Small preprint: one 7B model, A/B tasks only, single global coefficient in the simultaneous experiment,
no per-vector tuning. Treat as independent encouragement plus vocabulary, not strong precedent.
