# Results synthesis — working document

Workflow: Nimun reads the notebooks, we collect major findings per behaviour here,
then write Results/Discussion from this file. Findings go in as short claims, each
with its number and source. Nothing here is final prose.

Reference numbers below were audited 2026-08-16 against `exported_probes/` and
notebook outputs — safe to cite without re-checking.

---

## Deception

### Major findings
- (to fill)

### Verified reference numbers
- Detection ID: MM(last) 0.948, MM(mean) 1.000, LR(last) 0.996, LR(mean) 1.000, attention 1.000
- AI Liar OOD: MM(last) 0.959, MM(mean) 0.992, LR(last) 0.818, LR(mean) 0.974, attention 0.989
- Roleplaying OOD: MM(last) 0.669, MM(mean) 0.654, LR(last) 0.554, LR(mean) 0.706, attention 0.657
- Baseline readout gaps: ID +0.058 (degenerate; PD+ −0.906, PD− −0.964), AI Liar +1.052, Roleplaying +0.740
- Steering NIE at +1 / +2: ID MM 0.30/1.53, attn 0.20/0.56, LR −0.01/0.13;
  AI Liar MM 0.005/0.046, LR −0.047/−0.087, attn −0.000/0.010;
  Roleplaying MM 0.134/0.349, attn 0.088/0.214, LR −0.017/0.043
- Judge (AI Liar): behaviour max move 1.42 (LR at +2); coherence rises under MM at +1 (5.08→8.71)

## High-stakes

### Major findings
- (to fill)

### Verified reference numbers
- Detection ID: MM(last) 0.885, MM(mean) 0.962, LR(last) 0.968, LR(mean) 0.993, attention 0.996
- Anthropic-HH OOD: MM(last) 0.687, MM(mean) 0.737, LR(last) 0.686, LR(mean) 0.774, attention 0.755
- Steering NIE, OOD: MM 0.230 / 0.722 / 1.339 at +0.5/+1/+2; LR 0.120 at +1; attn 0.246 at +1
- Judge ID (MM): behaviour 2.46 at −2 → 4.08 at 0 → 5.88 at +1 → 7.54 at +2 (LR at +1: 5.00)
- Judge OOD (MM at +2): behaviour 1.67→2.38, coherence 6.21→3.92 (readout moves, behaviour doesn't)

## Sycophancy

### Major findings
- (to fill)

### Verified reference numbers
- Detection ID: MM(last) 0.976, MM(mean) 0.928, LR(last) 0.975, LR(mean) 0.970, attention 0.974
- Math OOD: MM(last) 0.879, MM(mean) 0.802, LR(last) 0.847, LR(mean) 0.825, attention 0.861
  (last beats mean for both estimators — only behaviour where this happens)
- Steering NIE: attention monotone (ID 0.16/0.36/0.81, math 0.25/0.54/0.77 at +0.5/+1/+2);
  MM non-monotone, dips negative at +1 (ID −0.53, math −0.92), partial recovery at +2 (0.69, 0.40)
- Judge: only sycophantic generations in the whole sweep are MM at +1 (ID 9.71, math 9.42),
  under degraded coherence (2.88, 4.17). Attention behaviour stays 1.0–1.2 on the POSITIVE half
  (rises to 1.42 ID / 2.00 math at −2 — do not claim "whole sweep")
- Attention-probe weights: echoed country token 0.38 (max) vs high-stakes diffuse max 0.03
- 79% of probed span is byte-identical prefix within a matched train pair; GA = 29% of train negatives

## Cross-behaviour / synthesis

### Major findings
- (to fill)

### Verified reference numbers
- Config: detection layers 12/20/19; steering blocks 11–15/6–10/6–10
- Direction cosines (cross-family): deception 0.54–0.81, high-stakes 0.26–0.53,
  sycophancy MM–LR max 0.135 (figure displays 0.13); all five sycophancy probes ≥0.802 OOD
- Best detector / best steerer (at +2) per held-out set:
  AI Liar MM(mean) 0.992 / MM +0.05 (uninformative);
  Roleplaying LR(mean) 0.706 / MM +0.35 (holds);
  HH LR(mean) 0.774 / MM +1.34 (holds);
  math MM(last) 0.879 / attention +0.77 (inverts)

## Discussion seeds
- (to fill)

---

## Known text issues to fix during the rewrite (from the 2026-08-16 audit)
1. "1.0–1.2 across the whole sweep" (sycophancy attention judge) — false at negative coefficients.
2. "at most 1.4 points, and both instruments together" (deception) — coherence moves +3.6;
   same paragraph has the leftover "dividing by that using it as the denominator".
3. "roughly seventeenfold" — 1.052/0.058 = 18.1×.
4. "≤0.13" cosine — raw 0.135; match the figure or reword.
5. "dose-response" wording (line ~1022) — Nimun dislikes "dose".
6. "cells" used in Results but the grid vocabulary is never introduced in Methodology.
