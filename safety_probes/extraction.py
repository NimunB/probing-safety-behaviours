"""Activation extraction and the disk cache (notebook §1b/§1c).

Two extractors, complementary in shape:

  * `get_pooled` / `extract_pooled_all_layers` — ALL layers, sequence collapsed.
    2 x [n, N_LAYERS, d_model]. Cheap; this is what makes the full layer sweep free.
  * `extract_full_sequence_one_layer` — ONE layer, every token kept. [n, max_length, d_model].
    ~8x bigger per layer despite being a single layer, which is why layer selection runs on
    pooled activations FIRST and this runs only at the layer you chose.

CACHE CAVEAT (unchanged from the notebook, deliberately)
--------------------------------------------------------
The key is (kind, model, name, n_examples, max_length) — it is CONTENT-BLIND. It does not hash
the texts or masks. Edit any data loader without changing `name` or the example count and the
cache will silently serve you activations for the old data. After editing a loader, delete the
matching `activation_cache/*.pt`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch as t
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

from . import runtime


def cache_dir() -> Path:
    """`<project_root>/activation_cache`, created on first use."""
    d = runtime.PROJECT_ROOT / "activation_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(*parts: str) -> str:
    """Stable content-hash key for the activation cache."""
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:16]


def cached(key: str, fn):
    """Load tensors/objects from disk if cached, else compute with fn() and save.

    Existence-only check: if the file is there the model never runs. Deleting a cache file is
    always safe — it costs a recompute, nothing more.
    """
    path = cache_dir() / f"{key}.pt"
    if path.exists():
        return t.load(path, weights_only=False)
    result = fn()
    t.save(result, path)
    return result


@t.no_grad()
def extract_pooled_all_layers(
    texts: list[str],
    det_masks: list[Bool[Tensor, " seq"]],
    batch_size: int = 16,   # was 8 (sized for the 24 GB card); 16 fits the 32 GB card easily (#52c)
    max_length: int = 512,
    desc: str = "extracting",
) -> tuple[Float[Tensor, "n layers d"], Float[Tensor, "n layers d"]]:
    """
    One forward pass per batch, pooled activations at EVERY layer — this is what makes the full
    layer sweep cheap. Returns (mean_pooled, last_token), each [n, N_LAYERS, d_model], fp16 on CPU.
    Pooling is over each example's detection mask (truncated to max_length if needed).
    """
    model, tokenizer = runtime.get_model(), runtime.get_tokenizer()
    means, lasts = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i : i + batch_size]
        batch_masks = det_masks[i : i + batch_size]

        # padding=True pads dynamically, to the longest text in THIS batch (not max_length).
        # That's fine here because we pool away the seq dimension before storing anything —
        # contrast extract_full_sequence_one_layer, which must pad every batch to a fixed length.
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(model.device)

        # out.hidden_states = tuple of (N_LAYERS + 1) tensors, each [b, seq, d];
        # hidden_states[0] is the embedding output, hidden_states[l+1] is transformer layer l.
        out = model(**inputs, output_hidden_states=True)

        # This [b, N_LAYERS, seq, d] block is the memory-heavy step (why batch_size=8):
        # it only lives on the GPU for this batch, then we pool it to [N_LAYERS, d] per example.
        hidden = t.stack(out.hidden_states[1:], dim=1)  # [b, N_LAYERS, seq, d]
        seq_len = hidden.shape[2]  # = longest text in this batch (dynamic padding)

        for j, m in enumerate(batch_masks):
            # Masks are built per-example (own length); the batch is right-padded to its longest
            # text. Truncate or False-pad each mask to seq_len so boolean indexing lines up.
            m = m[:seq_len]
            if len(m) < seq_len:
                m = t.cat([m, t.zeros(seq_len - len(m), dtype=t.bool)])  # padding -> False
            m = m.to(hidden.device)
            if m.sum() == 0:  # mask fully truncated away — fall back to last real token
                m = t.zeros(seq_len, dtype=t.bool, device=hidden.device)
                m[inputs["attention_mask"][j].sum() - 1] = True

            sel = hidden[j][:, m, :]                    # [N_LAYERS, n_masked, d]
            means.append(sel.mean(dim=1).half().cpu())  # mean pooling      -> [N_LAYERS, d]
            lasts.append(sel[:, -1, :].half().cpu())    # last masked token -> [N_LAYERS, d]
            # .half() = fp16: halves cache size; probes upcast to float32 when they train.

    return t.stack(means), t.stack(lasts)


def get_pooled(name, texts, masks, max_length=512):
    """Cached (mean_pooled, last_token) pooled activations, each [n, N_LAYERS, d_model]."""
    key = cache_key("pooled", runtime.MODEL_NAME, name, str(len(texts)), str(max_length))
    return cached(key, lambda: extract_pooled_all_layers(texts, masks, max_length=max_length, desc=name))


def extract_full_sequence_one_layer(name, texts, masks, layer, max_length=512, use_cache=True):
    """FULL token grid (un-pooled) at ONE layer, for the probes that need per-token activations
    (attention probe, per-token LR). Returns (acts [n, max_length, d], fmask [n, max_length]).

    use_cache=False computes in RAM only. Disk cost if cached is cumulative across layers:
    len(block) x n x max_length x d x 2 bytes — for a 5-layer block on the high-stakes train
    set that is ~63 GB, which is why `build_directions` extracts RAM-only. RAM cost is a
    per-layer PEAK (freed between layers), not a total.
    """
    model, tokenizer = runtime.get_model(), runtime.get_tokenizer()
    key = cache_key("full1layer", runtime.MODEL_NAME, name, str(layer), str(len(texts)), str(max_length))

    def compute():
        acts, msk = [], []
        for i in tqdm(range(0, len(texts), 8), desc=f"{name} L{layer}"):
            bt = texts[i:i + 8]
            bm = masks[i:i + 8]
            # padding="max_length" pads every batch to the SAME fixed length (unlike the dynamic
            # padding in extract_pooled_all_layers): we keep the seq dimension here, and
            # t.cat across batches below requires all batches to share it.
            inp = tokenizer(bt, return_tensors="pt", padding="max_length", truncation=True,
                            max_length=max_length).to(model.device)
            with t.no_grad():
                out = model(**inp, output_hidden_states=True)
            # Keep the FULL token grid, but only at the one chosen layer: [b, max_length, d].
            acts.append(out.hidden_states[layer + 1].half().cpu())
            attn = inp["attention_mask"].bool().cpu()  # [b, max_length], True = real token
            # Return the per-example DETECTION masks, not attention_mask (bug fixed 2026-07-10,
            # plan #10: `masks` was accepted but ignored). These fmasks gate the attention probe's
            # softmax — with attention_mask, the deception probe could attend to the system prompt,
            # which literally states the honest/deceptive persona (= the label). If truncation
            # removed every masked position, fall back to the last real token (an all-False row
            # would make the attention softmax NaN).
            for j, m in enumerate(bm):
                row = t.zeros(max_length, dtype=t.bool)
                row[: min(len(m), max_length)] = m[:max_length].bool()
                if not row.any():
                    row[attn[j].sum() - 1] = True
                msk.append(row)
        return t.cat(acts), t.stack(msk)

    return cached(key, compute) if use_cache else compute()
