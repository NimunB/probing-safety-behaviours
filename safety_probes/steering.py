"""Steering: hooks, calibration, the NIE readout, and direction-building (notebook §1g).

Covers everything on the WRITE side of the pipeline:
  * apply steering      — SteeringHook, generate_steered, compose_steering
  * calibrate a vector  — calibrate (coef 1.0 = one class-mean gap)
  * the READOUT         — readout_pdiff, nie_baseline_gap (the NIE measurement apparatus)
  * WHERE to steer      — nie_layer_sweep, pick_block
  * WHAT to steer with  — build_directions, calibrate_block (produce the exported vectors)

`steering_nie_llm_evaluation` (which couples this to the LLM judge) lives in `evaluation.py` to
keep the judge dependency out of this module.

CALIBRATION CONVENTION
----------------------
Vectors are pre-calibrated by `calibrate()` so that **coefficient 1.0 crosses the class-mean
gap** [GoT §6.1]. That scale is baked into the vector, not carried alongside it — which is why
`export.py` insists on exporting `calibrate_block()` output rather than raw directions. One
convention for MM / LR / LR-attention alike, so coefficients are comparable across methods.

ADDITION-ONLY BY DESIGN (decision 2026-07-11, plan #12)
-------------------------------------------------------
Every steering source we follow — [CAA], [GoT §6], van der Weij et al. (2024), and [NB]'s hooks —
steers by signed addition; a negative coefficient is subtraction. If refusal is ever added:
Arditi et al. SUPPRESS refusal by directional *ablation* (projection), which addition does not
replicate. Re-introduce an ablate mode then, and do not read a null negative-add as "not causal".
"""

from __future__ import annotations

import numpy as np
import torch as t
from torch import Tensor
from tqdm import tqdm

from . import runtime
from .extraction import extract_full_sequence_one_layer
from .probes import LRProbe, MMProbe, train_attention_probe
from .spec import BehaviourSpec


# WHERE the steering vector is written — HARD-CODED conventions, one per instrument (plan #59;
# the mode flags that briefly existed here were removed on the #12 precedent: no untested branches
# in the class that mutates the residual stream).
#
# READOUT (transcript scope): write ONLY to the item transcript's tokens — not the few-shot
#   demonstrations and not the trailing question. The demonstrations contain the literal
#   " YES"/" NO" tokens the NIE reads, so steering them would be direct pressure on the
#   instrument; [GoT §6.1] excludes the readout position for exactly this reason.
#
# GENERATION (CAA-exact scope): write to the FINAL PROMPT POSITION during prefill plus every
#   generated token. Verified against CAA-main's code (utils/helpers.py, quoted in
#   docs-for-nimun/sketchbook.md): `find_instruction_end_postion` returns the LAST token of
#   `[/INST]` — a prompt position — and `add_vector_from_position` masks `position_ids >=
#   from_pos`, INCLUSIVE. The final prompt position is the state that picks the first generated
#   token; skipping it (the short-lived "completion-only" scope, 2026-08-02..05) left the reply's
#   first word unsteered and flattened every judged dose-response (see plan #59 for the
#   three-regime evidence and the reproduce-instructions for the two historical scopes).


class SteeringHook:
    """ADDS a per-layer steering vector to a set of layers, over a configurable token scope.

    Scope is set by the *instrument*, not the caller: `readout_pdiff` narrows to the transcript
    span and the generation helpers apply the CAA-exact scope (see the module note above). The
    defaults here are all-token, so `compose_steering`, `ui_probe.py` and any exported bundle
    behave exactly as before unless a consumer opts in.

    All-token application remains the convention *within* a scope, uniformly across MM, LR and
    LR-attention, so a difference between arms can only come from the direction and never from one
    method touching fewer tokens than another (plan #33).

    Use as a context manager, or call enable()/disable() explicitly:

        with SteeringHook(layer_vectors, coef=1.0):
            out = model.generate(...)
    """

    def __init__(self, layer_vectors: dict[int, Tensor], coef: float,
                 positions: Tensor | None = None):
        self.layer_vectors = layer_vectors  # {layer: calibrated vector in raw activation space}
        self.coef = coef
        #: [batch, seq] bool — True where the vector is written. None = every position.
        #: Set per batch by `readout_pdiff`; the seq dim MUST match the forward pass.
        self.positions = positions
        #: INTERNAL routing, set by `_generation_scope` only: True while inside a generate() call,
        #: where prefill steers the final prompt position only (CAA-exact — see module note).
        self.in_generation = False
        self._handles = []

    def _make_fn(self, vec):
        v = vec.to(runtime.device)

        def fn(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            # Decode steps have seq len 1 (KV cache); prefill has the whole prompt.
            is_prefill = hs.shape[1] != 1
            add = self.coef * v.to(hs.dtype)
            if self.positions is not None and is_prefill:
                if self.positions.shape[1] != hs.shape[1]:
                    # Fail loud (#38): silently steering everything is the defect this scope fixes.
                    raise RuntimeError(
                        f"SteeringHook.positions has seq={self.positions.shape[1]} but the forward "
                        f"pass has seq={hs.shape[1]}. The mask must be rebuilt per batch.")
                m = self.positions.to(hs.device).unsqueeze(-1).to(hs.dtype)
                hs = hs + add * m
            elif self.in_generation and is_prefill:
                # CAA-exact (#59): steer ONLY the final prompt position — the state that picks the
                # first generated token (CAA-main masks `position_ids >= [/INST]-end`, inclusive).
                # Left-padded batches put every row's true last prompt token in the last column.
                last = t.zeros(hs.shape[1], 1, dtype=hs.dtype, device=hs.device)
                last[-1] = 1
                hs = hs + add * last
            else:
                hs = hs + add
            return (hs,) + output[1:] if isinstance(output, tuple) else hs

        return fn

    def enable(self):
        model = runtime.get_model()
        self._handles = [model.model.layers[l].register_forward_hook(self._make_fn(v))
                         for l, v in self.layer_vectors.items()]
        return self._handles

    def disable(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *exc):
        self.disable()
        return False


def compose_steering(*specs: tuple[dict[int, Tensor], float]) -> SteeringHook:
    """Combine several (layer_vectors, coef) pairs into ONE hook, summing where layers collide.

    ⚠️ RESEARCH-GRADE, NOT A VALIDATED METHOD (plan: UI §4.4). Whether two behaviour directions
    compose, cancel, or simply produce incoherence is an OPEN QUESTION that neither [CAA] nor
    [GoT] answers. This function makes the experiment *runnable*; it does not make the result
    *interpretable*. Treat any output as a pilot observation, not a measurement, until it has been
    through the same NIE + judge-coherence protocol as single-behaviour steering.

    Why one hook rather than two: stacking two forward hooks on the same layer also works
    (they chain additively), but summing first makes the applied vector explicit and inspectable,
    and keeps the semantics identical whether the blocks overlap or not.

    Note for the current bundles: deception steers [15-19] and high-stakes [9-13] — DISJOINT, so
    no summing actually occurs and the two behaviours act at different depths. That is a fact
    about these fitted blocks, not a guarantee; a future behaviour could overlap either.
    """
    merged: dict[int, Tensor] = {}
    for layer_vectors, coef in specs:
        for layer, vec in layer_vectors.items():
            contrib = coef * vec.float()
            merged[layer] = merged[layer] + contrib if layer in merged else contrib
    # Coefficient is folded into the vectors above, so the hook applies them at unit strength.
    return SteeringHook(merged, coef=1.0)


def calibrate(direction: Tensor, mean_acts_layer: Tensor, labels: Tensor) -> Tensor:
    """Scale a direction so coef 1.0 maps the mean negative onto the mean positive [GoT §6.1]:
    v = dir_hat * ((mu+ - mu-) . dir_hat). For the MM direction this returns (mu+ - mu-) itself."""
    d = direction.float()
    d_hat = d / (d.norm() + 1e-8)
    gap = ((mean_acts_layer[labels == 1].float().mean(0)
            - mean_acts_layer[labels == 0].float().mean(0)) @ d_hat)
    return d_hat * gap


class _generation_scope:
    """Mark a hook as inside a generate() call (CAA-exact prefill behavior), then restore.

    Hooks are reused across coefficients and by `nie_layer_sweep`, so the flag must not leak out of
    the generation path — the readout must keep seeing an unmodified hook.
    """

    def __init__(self, hook):
        self.hook, self.prev = hook, None

    def __enter__(self):
        if self.hook is not None:
            self.prev = self.hook.in_generation
            self.hook.in_generation = True
        return self.hook

    def __exit__(self, *exc):
        if self.hook is not None:
            self.hook.in_generation = self.prev
        return False


@t.no_grad()
def generate_steered(prompt_text: str, hook: SteeringHook | None, max_new_tokens: int = 80) -> str:
    """Greedy generation with an optional steering hook active. Deterministic (do_sample=False),
    so a coefficient of 0 reproduces the baseline exactly.

    STEERING SCOPE (#59, CAA-exact, hard-coded): the final prompt position is steered during
    prefill and every generated token during decode — matching CAA-main's inclusive
    `position_ids >= [/INST]-end` mask (verified from their utils/helpers.py).
    """
    model, tokenizer = runtime.get_model(), runtime.get_tokenizer()
    inp = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with _generation_scope(hook):
        handles = hook.enable() if hook is not None else []
        try:
            gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        finally:
            for h in handles:
                h.remove()
    return tokenizer.decode(gen[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@t.no_grad()
def generate_steered_batch(prompt_texts: list[str], hook: SteeringHook | None,
                           max_new_tokens: int = 80, batch_size: int = 8) -> list[str]:
    """Batched greedy generation with an optional steering hook (plan #52b).

    Same parameters as `generate_steered` (greedy, max_length=512 truncation, eos padding), but
    processes prompts in batches so the GPU is actually used. Prompts are LEFT-padded per batch so
    generation continues from the real last token; `padding_side` is flipped locally and restored
    (runtime pins it "right" for the mask/indexing paths — do not change that global). Caveat:
    batch padding can shift decode-side floating point, so generations may differ slightly in
    wording from the one-at-a-time path — acceptable for the judge, which has no cross-run
    comparability requirement (#44/#47).

    STEERING SCOPE (#59, CAA-exact, hard-coded): as `generate_steered`. Left padding makes the
    last column every row's true final prompt token, so the prefill add lands correctly for the
    whole batch; pad positions are never touched."""
    model, tokenizer = runtime.get_model(), runtime.get_tokenizer()
    out: list[str] = []
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        with _generation_scope(hook):
            handles = hook.enable() if hook is not None else []
            try:
                for i in range(0, len(prompt_texts), batch_size):
                    inp = tokenizer(prompt_texts[i:i + batch_size], return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(model.device)
                    gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
                    n_inp = inp["input_ids"].shape[1]
                    out.extend(tokenizer.decode(g[n_inp:], skip_special_tokens=True).strip() for g in gen)
            finally:
                for h in handles:
                    h.remove()
    finally:
        tokenizer.padding_side = old_side
    return out


# =========================================================================================
# The READOUT — our measurement apparatus (notebook §1g).
# =========================================================================================

def target_token_id(s: str) -> int:
    """First token id of a target string like " YES" — the token whose probability the readout reads."""
    tokenizer = runtime.get_tokenizer()
    return tokenizer(s, add_special_tokens=False)["input_ids"][0]


def render_transcript(spec: BehaviourSpec, item_text: str) -> str:
    """Plain-text rendering of an example for the readout prompt."""
    if spec.mask_policy == "assistant_only":
        return f"Dialogue:\n{item_text}\n"           # item_text already "User: ...\nAssistant: ..."
    return f"Message: {item_text}\n"                   # high-stakes: raw user message


def _transcript_char_spans(spec: BehaviourSpec, item_texts: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """Build the readout prompts and, for each, the [start, end) CHARACTER span of the item
    transcript inside it. Computed by construction (not by searching the prompt) so it is exact:
    the transcript sits at len(few_shot) + its offset within the rendered block."""
    prompts, spans = [], []
    base = len(spec.readout_few_shot)
    for x in item_texts:
        rendered = render_transcript(spec, x)
        start = base + rendered.index(x)          # skips "Dialogue:\n" / "Message: " wrapper
        prompts.append(spec.readout_few_shot + rendered + spec.readout_question)
        spans.append((start, start + len(x)))
    return prompts, spans


def _span_mask(enc, spans: list[tuple[int, int]], seq_len: int) -> Tensor:
    """[batch, seq] bool mask over the transcript tokens of each row, via char_to_token."""
    mask = t.zeros(len(spans), seq_len, dtype=t.bool)
    for j, (c0, c1) in enumerate(spans):
        st = enc.char_to_token(j, c0)
        if st is None:
            continue                              # transcript truncated away entirely -> steer nothing
        et = enc.char_to_token(j, c1 - 1)
        mask[j, st:(et + 1) if et is not None else seq_len] = True
    return mask


@t.no_grad()
def readout_pdiff(spec: BehaviourSpec, item_texts: list[str], steer_hook=None, batch_size=8) -> np.ndarray:
    """P(YES) - P(NO) at the final position of the few-shot readout, per example.

    You can't ask a residual stream "is this deceptive?" — instead we build a constrained prompt
    (few-shot examples + the transcript + "Was the assistant deceptive? Answer YES or NO.
    Answer:") and READ the model's judgment OUT of the next-token probabilities: PD = P(" YES") -
    P(" NO") at the final position. One forward pass, two logits, no generation. If a steering hook
    is active during that pass and the steered direction really carries the behaviour, the model's
    own YES/NO judgment shifts — that shift is what NIE measures.

    STEERING SCOPE (transcript, hard-coded — #56/#59): the hook is narrowed, per batch, to the
    transcript's own tokens — the few-shot demonstrations and the trailing question are NOT
    steered. See the module scope note for why this matters to the metric's validity.
    """
    model, tokenizer = runtime.get_model(), runtime.get_tokenizer()
    yes_id, no_id = target_token_id(spec.readout_targets[0]), target_token_id(spec.readout_targets[1])
    prompts, spans = _transcript_char_spans(spec, item_texts)
    narrow = steer_hook is not None
    out = []
    handles = steer_hook.enable() if steer_hook is not None else []
    try:
        for i in range(0, len(prompts), batch_size):
            inp = tokenizer(prompts[i:i+batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=1024).to(model.device)
            if narrow:
                # Rebuilt every batch: dynamic padding means seq len differs batch to batch, and
                # the hook raises rather than mis-apply a stale mask.
                steer_hook.positions = _span_mask(inp, spans[i:i+batch_size],
                                                  inp["input_ids"].shape[1])
            logits = model(**inp).logits
            last = inp["attention_mask"].sum(1) - 1
            probs = logits[t.arange(len(last)), last].softmax(-1)
            out.append((probs[:, yes_id] - probs[:, no_id]).cpu().float().numpy())
    finally:
        for h in handles:
            h.remove()
        if narrow:
            steer_hook.positions = None           # leave the hook reusable
    return np.concatenate(out)


def nie_baseline_gap(spec: BehaviourSpec, pos_texts: list[str], neg_texts: list[str]) -> tuple[float, float]:
    """Baseline control for any NIE: how far apart the classes ALREADY score in the YES/NO readout,
    before any steering. Returns (gap, base_neg) = (PD+ - PD-, mean PD of the negatives).

    The baseline = the unsteered model (coef 0, no hook). It measures the ruler NIE is expressed
    in: with nothing manipulated, how much more YES does the readout give the true positives than
    the true negatives? NIE then says "steering moved the negatives X% of that natural class
    distance." Every NIE divides by the gap measured on ITS OWN texts — a control belongs to the
    data it was measured on (§2f.i sweeps use train texts; §2f.iii/3f.iii evals use ID/OOD texts).
    """
    pd_pos = readout_pdiff(spec, pos_texts)
    pd_neg = readout_pdiff(spec, neg_texts)
    gap = float(pd_pos.mean() - pd_neg.mean())
    print(f"[{spec.name}] baseline readout gap PD+ - PD- = {gap:+.3f} "
          f"(PD+={pd_pos.mean():+.3f}, PD-={pd_neg.mean():+.3f})")
    if gap < 0.05:  # one-sided on purpose: catches BOTH "too small to divide by" and an inverted readout
        print("  ** WARNING: baseline gap ~ 0 or inverted -> NIE is ill-defined on these texts; the "
              "model can't self-report this behaviour in the readout format. Interpret with caution. **")
    return gap, float(pd_neg.mean())


# =========================================================================================
# WHERE to steer — per-layer NIE sweep and block selection (notebook §1g / §2f.i).
# =========================================================================================

def nie_layer_sweep(spec, train_mean, labels, sweep_pos_texts, sweep_neg_texts, layers=None):
    """WHERE to steer: the per-layer causal effect of the MM direction.
    For each layer: take that layer's mean-gap-calibrated MM direction, add it (coef +1) at that
    single layer while the readout runs on the negative texts, and record how far the negatives'
    readout moved toward the positive class, in units of the baseline gap (= NIE)."""
    layers = layers or list(range(runtime.N_LAYERS))
    # Control measurement on the sweep texts themselves (train split — see §2f.i note).
    gap, base_neg = nie_baseline_gap(spec, sweep_pos_texts, sweep_neg_texts)
    nie = []
    for layer in tqdm(layers, desc=f"{spec.name} NIE sweep"):
        # Layer-local MM direction, calibrated so coef 1.0 = cross the class-mean gap at THIS layer.
        vec = calibrate(MMProbe.from_data(train_mean[:, layer, :], labels).direction,
                        train_mean[:, layer, :], labels)
        hook = SteeringHook({layer: vec}, coef=1.0)
        steered = readout_pdiff(spec, sweep_neg_texts, steer_hook=hook).mean()
        nie.append((steered - base_neg) / (gap + 1e-8))
    return np.array(nie), gap


def pick_block(nie, width=5):
    """Contiguous window of `width` layers with the highest summed NIE."""
    best_s, best_i = -np.inf, 0
    for i in range(len(nie) - width + 1):
        s = nie[i:i+width].sum()
        if s > best_s:
            best_s, best_i = s, i
    return list(range(best_i, best_i + width))


# =========================================================================================
# WHAT to steer with — build and calibrate the exported direction set (notebook §1g / §2f.ii).
# =========================================================================================

def build_directions(spec, block, train_mean, labels, train_texts, train_masks, max_length):
    """Return {method: {layer: raw (uncalibrated) direction}} for MM, LR, LR-attention — ALL
    per-layer (decision (c), 2026-07-19: fully symmetric arms, CAA removed — plan #27).
    MM/LR are refit from the pooled cache at each block layer. LR-attention requires full-sequence
    activations AT EACH BLOCK LAYER: extracted here RAM-only (use_cache=False; caching would cost
    ~len(block) x the train set on disk) and freed after training. 3 seeds per layer; the direction
    = the mean of the seeds' W_out separators (calibrate() re-orients sign per layer regardless).
    Runtime: one full train-set forward pass + 3 attention-probe trainings PER BLOCK LAYER
    (~2 min/layer deception; ~8-12 min/layer high-stakes — the longest cell in the section)."""
    dirs = {"MM": {}, "LR": {}, "LR-attention": {}}
    for layer in block:
        X = train_mean[:, layer, :]
        dirs["MM"][layer] = MMProbe.from_data(X, labels).direction
        dirs["LR"][layer] = LRProbe.from_data(X, labels).direction
    for layer in tqdm(block, desc=f"{spec.name}: attention probes per block layer"):
        acts, fmask = extract_full_sequence_one_layer(
            f"{spec.name}_block", train_texts, train_masks, layer, max_length, use_cache=False)
        seeds = [train_attention_probe(acts, fmask, labels, seed=s,
                                       desc=f"  L{layer} attn seed {s + 1}/3") for s in range(3)]
        dirs["LR-attention"][layer] = t.stack([p.W_out.data[:, 0] for p in seeds]).mean(0)
        del acts, fmask, seeds
    return dirs


def calibrate_block(raw_dirs, train_mean, labels):
    """Apply GoT mean-gap calibration per layer -> {method: {layer: calibrated vector}}."""
    return {
        method: {layer: calibrate(vec, train_mean[:, layer, :], labels)
                 for layer, vec in layer_dirs.items()}
        for method, layer_dirs in raw_dirs.items()
    }
