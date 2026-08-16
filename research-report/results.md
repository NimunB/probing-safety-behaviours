# Results synthesis — working document

Mirrors the report's new Results skeleton: per behaviour — Layer Selection /
Detection (Probing) / Steering; cross-behaviour material goes to Synthesis
(measured results) or Discussion (interpretation). Findings go in as short
claims. Reference numbers were audited 2026-08-16 against `exported_probes/`
and notebook outputs — safe to cite without re-checking.

The removed body text is backed up (scratchpad `results_prose_backup.tex` and
git history) if a phrasing is worth recovering.

---

## A. Deception

### A.1 Layer Selection
Findings:
-

Reference: detection layer 12 (argmax CV AUROC); steering block 11–15; the
block encloses the detection layer — only behaviour where read and write peak
in the same region.

### A.2 Detection (Probing)
Findings:
-

Reference: ID — MM(last) 0.948, MM(mean) 1.000, LR(last) 0.996, LR(mean) 1.000,
attention 1.000 (four of five ≥0.996, three at 1.000).
AI Liar — 0.959 / 0.992 / 0.818 / 0.974 / 0.989 (MM(mean) best, near-tie at ceiling).
Roleplaying — 0.669 / 0.654 / 0.554 / 0.706 / 0.657 (LR(mean) best; pooled 0.722).
AI Liar answers byte-identical across labels; roleplaying completions genuinely
differ — collapse happens on the *content-defined* set, opposite of a
surface-cue account. Apollo at 70B hold 0.964 on roleplaying (their lowest).

### A.3 Steering
Findings:
-

Reference: baseline gaps — ID +0.058 (degenerate: PD+ −0.906, PD− −0.964; both
classes answer NO; 1/0.058 ≈ 17×, 1.052/0.058 ≈ 18×), AI Liar +1.052,
Roleplaying +0.740. NIE at +1/+2: ID MM 0.30/1.53, attn 0.20/0.56, LR −0.01/0.13;
AI Liar MM 0.005/0.046, attn −0.000/0.010, LR −0.047/−0.087;
Roleplaying MM 0.134/0.349, attn 0.088/0.214, LR −0.017/0.043.
Judge (AI Liar): behaviour max move 1.42 (LR at +2); coherence RISES under MM
at +1 (5.08→8.71) — don't claim "both instruments move ≤1.4".

## B. High-stakes

### B.1 Layer Selection
Findings:
-

Reference: detection layer 20; steering block 6–10 — block well below the
detection layer.

### B.2 Detection (Probing)
Findings:
-

Reference: ID — 0.885 / 0.962 / 0.968 / 0.993 / 0.996 (attention best).
Anthropic-HH — 0.687 / 0.737 / 0.686 / 0.774 / 0.755 (LR(mean) best; last-token
cells at the bottom, 0.686–0.687, matching McKenzie's "consistently performs
poorly" for last-token).

### B.3 Steering
Findings:
-

Reference: OOD NIE (MM): 0.230 / 0.722 / 1.339 at +0.5/+1/+2 — monotone;
LR 0.120, attn 0.246 at +1. Judge ID (MM): 2.46 at −2 → 4.08 at 0 → 5.88 at +1
→ 7.54 at +2 (LR at +1: 5.00). OOD detail: MM at +2 moves readout a full class
gap (1.34) while judged behaviour barely moves (1.67→2.38) and coherence falls
(6.21→3.92) — concept written into representation without the conversation
becoming high-stakes.

## C. Sycophancy

### C.1 Layer Selection
Findings:
-

Reference: detection layer 19; steering block 6–10 — argmax thirteen layers
above the block start; widest read/write offset.

### C.2 Detection (Probing)
Findings:
-

Reference: ID — 0.976 / 0.928 / 0.975 / 0.970 / 0.974 (MM(last) best).
Math — 0.879 / 0.802 / 0.847 / 0.825 / 0.861 (MM(last) best). last beats mean
for BOTH estimators — only behaviour where this happens; predicted by the 79%
byte-identical templated prefix (measured over 505 matched train pairs).
Attention-probe weights: echoed country token 0.38 max vs high-stakes diffuse
0.03 max. GA = 29% of train negatives; label is relational.

### C.3 Steering
Findings:
-

Reference: attention monotone (ID 0.16/0.36/0.81; math 0.25/0.54/0.77 at
+0.5/+1/+2). MM non-monotone: dips negative at +1 (ID −0.53, math −0.92),
partial recovery at +2 (0.69, 0.40). Judge: only sycophantic generations in
the whole sweep are MM at +1 (ID 9.71, math 9.42) under degraded coherence
(2.88, 4.17). Attention judged behaviour stays 1.0–1.2 on the POSITIVE half
only (1.42 ID / 2.00 math at −2 — do not claim "whole sweep"). Judge and
readout disagree in SIGN for MM at +1.

## D. Synthesis (measured, cross-behaviour)
Findings:
-

Reference: Table II — best detector / best steerer at +2 per held-out set:
AI Liar MM(mean) 0.992 / MM +0.05 (uninformative); Roleplaying LR(mean) 0.706 /
MM +0.35 (holds); HH LR(mean) 0.774 / MM +1.34 (holds); math MM(last) 0.879 /
attention +0.77 (inverts).
Layer geometry: all causal bands in layers 6–15; detection at 12/20/19; CAA's
optimal layer 13 of 32 mid-network; ActAdd: "intervening at the middle layers
is most effective"; CAA transfer drop-off past layer 17.
Cosines (cross-family): deception 0.54–0.81, high-stakes 0.26–0.53, sycophancy
MM–LR max 0.135 (figure shows 0.13); all five sycophancy probes ≥0.802 OOD.
Marks ratio: MM NIE 0.77–0.90 vs LR 0.13–0.19 at 13B (4–7×); our HH 0.72 vs
0.12 at +1 — same ratio. Coherence at +2: MM degrades nearly every set (HH
6.21→3.92, math 8.17→1.00) while LR/attention stay near baseline (attention on
math dips to 5.42); AI Liar anomaly — coherence RISES under MM (5.08→7.04).

## Discussion seeds (interpretation)
-

---

## Known text issues — absorb during the rewrite (2026-08-16 audit)
1. Sycophancy attention judge: floor claim holds for positive half only.
2. Deception AI Liar: "at most 1.4 points" is behaviour-only; coherence moves
   +3.6. Drop the "dividing by that using it as" leftover phrasing.
3. "seventeenfold" → ~eighteenfold (1.052/0.058 = 18.1).
4. "≤0.13" cosine — raw 0.135; match the figure or reword.
5. Avoid "dose-response" (also still at Steering §III-D line ~713 and two
   appendix captions).
6. "cells" vocabulary — define the grid once or say "probes".
