"""Steering dose-response evaluation — couples the readout, generation, and the judge (§1g cell 38).

Its own module (not `steering.py`) so that the judge dependency stays out of the core steering
code: the UI and the parity harness can apply steering without importing `anthropic`. This is the
one place that ties WRITE-side steering to the LLM judge.
"""

from __future__ import annotations

import numpy as np

from concurrent.futures import ThreadPoolExecutor

from .judge import get_client, judge_generation
from .steering import SteeringHook, generate_steered_batch, nie_baseline_gap, readout_pdiff


def steering_nie_llm_evaluation(spec, dirs, prompt_texts, readout_texts, pos_readout_texts, coefs):
    """WHAT steering does: effect per direction (MM/LR/LR-attention) per coefficient — the
    dose-response experiment. Two independent measurements at every (direction, coef):
      1. NIE   - steer while the readout runs on `readout_texts` (known negatives); record the shift
                 toward the positive class in units of the baseline gap [GoT §6.1]. NIE 1.0 =
                 steering moved the negatives' readout one full class gap. No generation involved.
      2. judge - steer while GENERATING from `prompt_texts`; the LLM judge scores each generation
                 for the behaviour AND coherence (the guard against "steering that works by breaking
                 the model"). No ANTHROPIC_API_KEY -> judge intentionally skipped, columns NaN,
                 NIE still fills. Passing `prompt_texts=[]` also skips the judge deliberately —
                 for NIE-only eval sets (deception ID, plan #46). But when the judge IS configured
                 and prompts were given it must not fail quietly: a point scored on fewer than the
                 requested generations prints a visible shortfall warning, and a point that scores
                 0/N raises (systematic breakage -> stop loudly, no empty panels).
    Returns {method: {coef, nie, judge_beh, judge_coh}} - feed to plot_steering_effects."""
    results = {m: {"coef": [], "nie": [], "judge_beh": [], "judge_coh": []} for m in dirs}
    # Control measurement on THESE eval texts (a control belongs to the data it was measured on).
    gap, base_neg = nie_baseline_gap(spec, pos_readout_texts, readout_texts)
    denom = gap if abs(gap) > 1e-6 else 1e-6  # numerical guard only; the warning already fired above
    judge_on = get_client() is not None  # False => judge deliberately skipped (no key); NaN expected, not a failure

    def judged_generations(hook, label):
        if not prompt_texts or not judge_on:      # nothing to generate/judge (deliberate skips)
            return (np.nan, np.nan)
        # (#52b) batch-generate on the GPU, then judge the batch with parallel API calls —
        # replaces the interleaved one-at-a-time loop (GPU idled during every judge round-trip).
        gens = generate_steered_batch(prompt_texts, hook)
        beh, coh = [], []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for j in ex.map(lambda pg: judge_generation(spec, pg[0], pg[1]), zip(prompt_texts, gens)):
                if j:
                    beh.append(j["behaviour"]); coh.append(j["coherence"])
        n_ok, n_req = len(beh), len(prompt_texts)
        if judge_on and n_req:  # only police failures when judging was actually requested (#46)
            if n_ok == 0:
                raise RuntimeError(
                    f"Judge scored 0/{n_req} generations at {label}: systematic failure (bad model/key/"
                    "schema, not a transient blip). Stopping loudly instead of emitting empty/NaN judge "
                    "panels - see the 'judge error:' line(s) above for the cause.")
            if n_ok < n_req:
                print(f"  ⚠ [judge] {label}: only {n_ok}/{n_req} generations scored "
                      "(mean over survivors; treat this point as lower-n).")
        return (np.mean(beh) if beh else np.nan, np.mean(coh) if coh else np.nan)

    # coef 0 = no hook, deterministic decoding -> identical generations for every direction.
    # Measure the baseline point ONCE and reuse; its NIE is exactly 0 by definition.
    base_judge = judged_generations(None, "baseline (coef 0)") if any(float(c) == 0.0 for c in coefs) else None
    for method, layer_dirs in dirs.items():
        for coef in coefs:
            results[method]["coef"].append(coef)
            if float(coef) == 0.0:
                results[method]["nie"].append(0.0)
                results[method]["judge_beh"].append(base_judge[0])
                results[method]["judge_coh"].append(base_judge[1])
                continue
            hook = SteeringHook(layer_dirs, coef=coef)
            # (1) readout shift under steering
            pdiff = readout_pdiff(spec, readout_texts, steer_hook=hook)
            results[method]["nie"].append((pdiff.mean() - base_neg) / denom)
            # (2) N_JUDGE_PROMPTS judged generations
            jb, jc = judged_generations(hook, f"{method}@coef{float(coef):+g}")
            results[method]["judge_beh"].append(jb)
            results[method]["judge_coh"].append(jc)
    return results
