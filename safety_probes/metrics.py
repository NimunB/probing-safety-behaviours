"""Probe training, scoring, and evaluation metrics (notebook §1f).

Numeric only — every figure lives in `viz.py`, so this module imports no plotting library and
can be used headlessly.

Metric choices, and why:
  * **AUROC with bootstrap 95% CI** — threshold-free, and AI Liar is small enough that a point
    estimate alone is not defensible. [HS Fig. 2] uses 3 seeds + 95% CI.
  * **TPR@1%FPR** — [HS Fig. 3b] comparability. Returns NaN rather than a misleading number on
    eval sets too small for 1% FPR to be reachable (see `tpr_at_fpr`).
  * **Frozen-threshold accuracy: deliberately absent** (dropped 2026-07-19, plan #28) — it
    measured threshold transfer between datasets, not representation quality.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch as t
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from torch import Tensor

from . import runtime
from .probes import LRProbe, MMProbe, train_attention_probe

N_TOKEN_CAP = 60000  # max flattened token-rows for per-token LR training (memory/tractability)


# =========================================================================================
# Metrics
# =========================================================================================

def bootstrap_auroc(labels, scores, n_boot=1000):
    """AUROC with a bootstrap 95% CI. Resamples that lose a class are skipped, not counted."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    auroc = roc_auc_score(labels, scores)
    rng = np.random.RandomState(runtime.SEED)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(labels), len(labels))
        if len(set(labels[idx].tolist())) < 2:
            continue
        boots.append(roc_auc_score(labels[idx], scores[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (auroc, auroc)
    return auroc, lo, hi


def tpr_at_fpr(labels, scores, target_fpr=0.01):
    """TPR at the strictest threshold with FPR <= target_fpr within this dataset [HS Fig. 3b].

    Returns NaN when the set has too few negatives for target_fpr to correspond to >=1 tolerated
    false positive (n_neg < 1/target_fpr = 100 at 1%): 1% FPR is then UNREACHABLE and the metric
    would silently degrade to TPR@0FP. So AI Liar (27 neg) -> n/a; Roleplaying (371), the pool
    (398), high-stakes -> computed. (plan #18c, resolved via the roleplaying OOD set, #35.)
    """
    labels = np.asarray(labels)
    if int((labels == 0).sum()) < 1 / target_fpr:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


# =========================================================================================
# Probe grid
# =========================================================================================

def token_level_dataset(full_acts, token_masks, labels, cap=N_TOKEN_CAP):
    """Apollo per-token training set: every masked-token activation is one row, labelled by its
    example. full_acts: [n, seq, d] at the detection layer; token_masks: per-example bool masks."""
    rows, ys = [], []
    S = full_acts.shape[1]
    for j in range(full_acts.shape[0]):
        m = token_masks[j]
        L = min(len(m), S)
        sel = full_acts[j, :L][m[:L].bool()]
        if sel.shape[0] == 0:
            continue
        rows.append(sel.float())
        ys.append(t.full((sel.shape[0],), float(labels[j])))
    X, y = t.cat(rows), t.cat(ys)
    if X.shape[0] > cap:
        idx = t.randperm(X.shape[0], generator=t.Generator().manual_seed(runtime.SEED))[:cap]
        X, y = X[idx], y[idx]
    return X, y


def build_probe_grid(train_mean, train_last, train_masks_padded, train_acts_full, token_masks, labels):
    """Train the 5 probes at a single layer's activations. Returns {name: (probe, pooling)}.

    NOTE the `LR-attention` entry's probe is a LIST of 3 seed probes, not a single module — every
    consumer (`probe_scores`, `export_behaviour`, the cosine matrix) branches on that.
    """
    grid, _t0 = {}, time.time()

    def _done(label):
        print(f"  [{len(grid)}/5] {label:<13} done  ({time.time() - _t0:6.1f}s elapsed)")

    print(f"Building probe grid (5 probes), acts {tuple(train_acts_full.shape)}...")
    grid["MM (last)"] = (MMProbe.from_data(train_last, labels), "last");  _done("MM (last)")
    grid["MM (mean)"] = (MMProbe.from_data(train_mean, labels), "mean");  _done("MM (mean)")
    grid["LR (last)"] = (LRProbe.from_data(train_last, labels), "last");  _done("LR (last)")
    # LR (mean): Apollo per-token training. Eval (probe_scores pooling="mean") scores the
    # mean-pooled vector, which equals the mean of per-token scores — so only TRAINING changes,
    # not evaluation.
    Xtok, ytok = token_level_dataset(train_acts_full, token_masks, labels)
    print(f"  LR (mean) per-token training set: {Xtok.shape[0]} token-rows")
    grid["LR (mean)"] = (LRProbe.from_data(Xtok, ytok), "mean");          _done("LR (mean)")
    # Attention probe: 3 seeds, all kept for the CI.
    # Slowest step, now GPU-chunked (#52a): same full-batch gradient, chunks streamed to the GPU.
    print("  LR-attention: 3 seeds x 200 epochs, GPU-chunked full-batch training (#52a)...")
    attn_probes = [train_attention_probe(train_acts_full, train_masks_padded, labels, seed=s,
                                         desc=f"    LR-attention seed {s + 1}/3")
                   for s in range(3)]
    grid["LR-attention"] = (attn_probes, "attention");                    _done("LR-attention")
    print(f"Probe grid COMPLETE: {list(grid)}  ({time.time() - _t0:.1f}s total)")
    return grid


def probe_scores(probe, pooling, mean_acts, last_acts, full_acts=None, full_masks=None):
    """Uniform scoring interface across pooling types. Returns a numpy score vector."""
    if pooling == "mean":
        return probe.score(mean_acts).detach().numpy()
    if pooling == "last":
        return probe.score(last_acts).detach().numpy()
    if pooling == "attention":  # probe is a list of 3 seeds; average their sigmoid scores
        with t.no_grad():
            return np.mean([p(full_acts, full_masks).sigmoid().numpy() for p in probe], axis=0)
    raise ValueError(pooling)


def pack(mean, last, full, fmask, labels):
    """Bundle one eval set into the dict shape `evaluate_grid` expects."""
    return {"mean": mean, "last": last, "full": full, "fmask": fmask, "labels": labels}


def cat_pack(a, b):
    """Concatenate two eval packs (used for the pooled deception OOD column, plan #35)."""
    return pack(t.cat([a["mean"], b["mean"]]), t.cat([a["last"], b["last"]]),
                t.cat([a["full"], b["full"]]), t.cat([a["fmask"], b["fmask"]]),
                t.cat([a["labels"], b["labels"]]))


def evaluate_grid(grid, eval_packs):
    """AUROC (bootstrap 95% CI) + TPR@1%FPR per probe per eval set, as a DataFrame."""
    rows = []
    for probe_name, (probe, pooling) in grid.items():
        for eval_name, ev in eval_packs.items():
            sc = probe_scores(probe, pooling, ev["mean"], ev["last"], ev.get("full"), ev.get("fmask"))
            auroc, lo, hi = bootstrap_auroc(ev["labels"].numpy(), sc)
            tpr = tpr_at_fpr(ev["labels"].numpy(), sc)
            rows.append({"probe": probe_name, "eval": eval_name,
                         "AUROC": f"{auroc:.3f} [{lo:.2f},{hi:.2f}]",
                         "TPR@1%FPR": "n/a (<100 neg)" if np.isnan(tpr) else f"{tpr:.3f}"})
    return pd.DataFrame(rows)


# =========================================================================================
# Layer selection
# =========================================================================================

def cv_auroc_by_layer(mean_acts, labels, groups, n_splits=5, probe_cls=MMProbe):
    """Pair-aware 5-fold CV AUROC of a probe at each layer.

    `groups` = pair ids, so no pair is split across folds — splitting one would leak a
    near-duplicate example (deception: the same fact under the opposite persona; high-stakes: a
    tone/language variant of the same situation) from train into test and inflate the AUROC.

    probe_cls: MMProbe (closed-form, fast — the selector) or LRProbe (slow; robustness overlay).
    """
    n_layers = mean_acts.shape[1]
    gkf = GroupKFold(n_splits=n_splits)
    per_layer = np.zeros((n_layers, n_splits))
    for layer in range(n_layers):
        X = mean_acts[:, layer, :]
        for fold, (tr, te) in enumerate(gkf.split(X, labels, groups)):
            probe = probe_cls.from_data(X[tr], labels[tr])
            per_layer[layer, fold] = roc_auc_score(
                labels[te].numpy(), probe.score(X[te]).detach().numpy())
    return per_layer.mean(1), per_layer.std(1)
