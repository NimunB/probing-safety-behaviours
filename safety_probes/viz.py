"""All figures and heatmaps (notebook cells 17, 24, 26, 27, 28, 38, 39).

VISUALIZATION ONLY. Nothing here computes a reported number — the probe grid scores through
`metrics.probe_scores`, not through these paths; the heatmaps and panels re-score for display.
Kept out of `metrics.py`/`steering.py` so those stay headless (no plotly, no circuitsvis import).

COMPLETES THE ARENA CUT. `process_str_tokens`, `visualize_token_scores`, and
`score_tokens_with_probe` are vendored VERBATIM from ARENA's utils.py (only `utils.` prefixes and
notebook globals rewritten). After this module, nothing in the project imports the course repo.
`circuitsvis` is therefore a real dependency here (it was deferred while only the headless modules
existed — plan M7); pin it in requirements before running on a fresh instance.
"""

from __future__ import annotations

import html as _html
import re as _re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torch as t
from matplotlib.colors import to_hex
from IPython.display import HTML, display
from jaxtyping import Bool, Float
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor

from . import runtime
from .chat import clean_bpe_token  # noqa: F401  (kept importable from here too, as in the notebook)
from .metrics import probe_scores
from .steering import SteeringHook, generate_steered

STEER_COLORS = {"MM": "#636efa", "LR": "#00cc96", "LR-attention": "#FFA15A"}  # one color per method

# =========================================================================================
# Matplotlib figures (render as inline PNG -> visible on GitHub; publication-style).
#
# WHY MATPLOTLIB, NOT PLOTLY, for these: Plotly output is stored as vnd.plotly.v1+json, which
# GitHub cannot render — the figures show as blank space. matplotlib inline output IS a PNG, so
# it renders natively on GitHub with no kaleido dependency, and the seaborn-v0_8-darkgrid style
# (shipped with matplotlib, no seaborn install) matches the thesis reference aesthetic.
# =========================================================================================

MPL_STYLE = "seaborn-v0_8-darkgrid"     # ships with matplotlib; the seaborn look without seaborn
MM_COLOR, LR_COLOR = "#4c72b0", "#dd8452"   # seaborn "deep" blue / orange (probe identity)
POS_COLOR, NEG_COLOR = "#e8000b", "#023eff"  # seaborn "bright" red / blue — class identity: positive / negative
PROBE_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]  # seaborn "deep": one per probe
# Sequential map for attention weights (≥0): white (low) → dark green (high), matching the
# circuitsvis attention view (white → green).
ATTN_CMAP = plt.get_cmap("Greens")


def _use_style():
    """Apply the shared plotting style; safe to call repeatedly."""
    try:
        plt.style.use(MPL_STYLE)
    except OSError:                       # older matplotlib names it differently
        plt.style.use("seaborn-darkgrid")
    # Higher DPI so inline figures aren't upscaled-and-fuzzy on retina screens, and saved PNGs
    # (for GitHub) are crisp. For the sharpest NOTEBOOK display also set, once, in the notebook:
    #     %config InlineBackend.figure_format = "retina"
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200})


# =========================================================================================
# Report-figure auto-save (plan #59). Each notebook calls save_figures_to("<behaviour>") once;
# matplotlib plot functions then take save="<slot>" and write research-report/figures/
# <behaviour>/<slot>.png as a side effect of rendering, so every run refreshes the report's
# images. EXCEPTION: plot_steering_effects is plotly (PNG export would need kaleido + a headless
# Chrome — rejected 2026-08-05); the dose-response panels remain manual screenshots, dropped into
# the same per-behaviour folder under the slot names in research-report/figures/README.md.
# =========================================================================================

FIGURE_DIR: Path | None = None      # set by save_figures_to(); None = saving disabled (UI, tests)


def save_figures_to(behaviour: str) -> Path:
    """Route every subsequent save=-tagged figure to research-report/figures/<behaviour>/."""
    global FIGURE_DIR
    FIGURE_DIR = runtime.PROJECT_ROOT / "research-report" / "figures" / behaviour
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"report figures -> {FIGURE_DIR.relative_to(runtime.PROJECT_ROOT)}/")
    return FIGURE_DIR


def _slug(name: str) -> str:
    """Class name -> filename fragment ('High-stakes' -> 'high_stakes')."""
    return _re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _finish(fig, save: str | None):
    """Save-then-show for matplotlib figures. A save tag with no configured dir warns rather than
    silently dropping the figure — same fail-visible convention as the cache/provisioning guards."""
    if save:
        if FIGURE_DIR is None:
            print(f"  ** figure save '{save}' requested but save_figures_to() was never called — skipped **")
        else:
            out = FIGURE_DIR / f"{save}.png"
            fig.savefig(out, dpi=200, bbox_inches="tight")
            shown = out.relative_to(runtime.PROJECT_ROOT) if out.is_relative_to(runtime.PROJECT_ROOT) else out
            print(f"  [saved] {shown}")
    plt.show()


def plot_layer_selection(cv_mean, cv_std, lr_mean, lr_std, chosen_layer, behaviour, save=None):
    """Per-layer pair-aware CV AUROC: MM (the selector) + LR (robustness overlay), with the chosen
    detection layer marked. matplotlib so it renders on GitHub; y-axis floored at chance (0.5) so
    the curves fill the panel instead of floating in whitespace."""
    import numpy as np
    _use_style()
    xs = np.arange(len(cv_mean))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.errorbar(xs, cv_mean, yerr=cv_std, color=MM_COLOR, marker="o", ms=3, lw=1.8,
                capsize=2, label="MM (selector)", zorder=3)   # selector drawn on top of the overlay
    ax.errorbar(xs, lr_mean, yerr=lr_std, color=LR_COLOR, marker="o", ms=3, lw=1.8,
                capsize=2, label="LR (overlay)", zorder=2)
    ax.axvline(chosen_layer, color="#555", ls="--", lw=1.2, zorder=0)
    ax.annotate(f"Layer {chosen_layer}", xy=(chosen_layer, ax.get_ylim()[0]),
                xytext=(4, 6), textcoords="offset points", fontsize=10, color="#333")
    ax.set_xlabel("Layer"); ax.set_ylabel("5-fold CV AUROC")
    ax.set_ylim(0.5, 1.01)               # 0.5 = chance; removes the dead band below the data
    ax.set_xlim(-0.5, len(cv_mean) - 0.5)
    ax.set_title(f"{behaviour} detection layer (pair-aware CV) = {chosen_layer}\n"
                 f"AUROC {cv_mean[chosen_layer]:.3f}", fontsize=13)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()                   # kills the title bleed
    _finish(fig, save)


def plot_nie_bars(nie, steer_block, behaviour, save=None):
    """NIE (Natural Indirect Effect) for steering ONE layer at a time [GoT §3 group (b)]: the readout
    shift, in class-gap units, from adding the class-mean direction at that layer. The chosen steering
    block is highlighted, the rest muted. matplotlib so it renders on GitHub."""
    import numpy as np
    _use_style()
    xs = np.arange(len(nie))
    in_block = set(steer_block)
    colors = ["#c44e52" if i in in_block else "#bbb" for i in range(len(nie))]   # block red, rest grey
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(xs, nie, color=colors)
    ax.axhline(0, color="#888", lw=0.8, zorder=0)   # NIE can be negative
    ax.set_xlabel("Layer"); ax.set_ylabel("NIE (single-layer, coef +1)")
    ax.set_xlim(-0.6, len(nie) - 0.4)
    ax.set_title(f"{behaviour} NIE by layer", fontsize=12)
    fig.tight_layout()
    _finish(fig, save)


# =========================================================================================
# Vendored from ARENA utils.py — do not edit without a cache wipe + re-audit (see chat.py note).
# =========================================================================================

def process_str_tokens(str_tokens: list[str]) -> list[str]:
    """Sanitize special tokens so they render as text rather than HTML tags."""
    replacements = {
        "<bos>": "[bos]", "<s>": "[s]", "</s>": "[/s]",
        "<start_of_turn>": "[start_of_turn]", "<end_of_turn>": "[end_of_turn]", "\n": "\\n",
    }
    for k, v in replacements.items():
        str_tokens = [tok.replace(k, v) for tok in str_tokens]
    return str_tokens


def visualize_token_scores(
    str_tokens, scores, mask, title="", label="", score_value=None, centering_value=None,
    min_value=None, max_value=None, positive_color="red", negative_color="blue",
    show=True, filename=None, score_name="probe score",
) -> str:
    """Render per-token colored scores using circuitsvis. Vendored from ARENA utils.py."""
    from circuitsvis.tokens import colored_tokens

    if isinstance(scores, Tensor):
        scores = scores.cpu().tolist()
    if isinstance(mask, Tensor):
        mask = mask.cpu().tolist()

    assert len(str_tokens) == len(scores), f"Token count ({len(str_tokens)}) != score count ({len(scores)})"

    scores_unmasked = [s for s, m in zip(scores, mask) if m]
    scores = [s * m for s, m in zip(scores, mask)]
    if centering_value is not None:
        scores = [(s - centering_value) * m for s, m in zip(scores, mask)]

    std_dev = np.std(scores_unmasked)
    if std_dev > 0:
        scores = [s / std_dev for s in scores]

    if min_value is None:
        min_value = min(scores)
    if max_value is None:
        max_value = max(scores)
    max_value = max(max_value, -min_value)
    min_value = -max_value

    tokens_clean = process_str_tokens(str_tokens)
    scores_rounded = [round(s, 3) for s in scores]

    cv_html = str(colored_tokens(tokens_clean, scores_rounded, min_value=min_value,
                                 max_value=max_value, positive_color=positive_color,
                                 negative_color=negative_color)) + "</div>"

    header_parts = []
    if title:
        header_parts.append(f"<h3 style='font-family: Arial, sans-serif;'>{title}</h3>")
    if score_value is not None or label:
        subtitle_items = []
        if score_value is not None:
            subtitle_items.append(f"{score_name.capitalize()}: {score_value:.2f}")
        if label:
            label_color = {"Honest": "#4c72b0", "Deceptive": "#c44e52"}.get(label, "#666")
            subtitle_items.append(
                f"Ground truth label: <span style='color: {label_color}; font-weight: bold;'>{label}</span>")
        header_parts.append(
            f"<p style='font-family: Arial, sans-serif; font-size: 14px;'>{', '.join(subtitle_items)}</p>")

    html = "\n".join(header_parts) + cv_html
    if show:
        display(HTML(html))
    if filename:
        with open(filename, "w") as f:
            f.write(html)
    return html


def score_tokens_with_probe(text, model, tokenizer, probe_direction, layer, scaler=None):
    """Tokenize, extract activations at `layer`, score each token with a probe direction.
    Vendored from ARENA utils.py."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with t.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer + 1]  # [1, seq, d_model]
    acts = hidden[0].cpu().float()             # [seq, d_model]
    if scaler is not None:
        acts_scaled = t.tensor(scaler.transform(acts.numpy()), dtype=t.float32)
        scores = acts_scaled @ t.tensor(probe_direction, dtype=t.float32)
    else:
        scores = acts @ probe_direction.cpu().float()
    str_tokens = [tokenizer.decode(tok_id) for tok_id in inputs["input_ids"][0]]
    return str_tokens, scores


# =========================================================================================
# Masking / data sanity (cell 17)
# =========================================================================================

def show_masked_tokens(text: str, mask: Bool[Tensor, " seq"], title: str):
    tokenizer = runtime.get_tokenizer()
    ids = tokenizer(text)["input_ids"]
    toks = [clean_bpe_token(tokenizer.decode(i)) for i in ids]
    rendered = "".join(
        f"<span style='background:#fca'>{tok}</span>" if m else f"<span style='color:#999'>{tok}</span>"
        for tok, m in zip(toks, mask.tolist())
    )
    display(HTML(f"<h4>{title}</h4><div style='font-family:monospace; line-height:1.7'>{rendered}</div>"
                 f"<p>{int(mask.sum())}/{len(toks)} tokens probed (highlighted)</p>"))


# =========================================================================================
# Detection evaluations — ROC + PCA + heatmaps + cosine matrix (cells 24, 26, 27, 28)
# =========================================================================================

def plot_roc_curves(grid, ev, title, save=None):
    """ROC curves for every probe on one eval pack ([NB] §5-style). Area under each curve = the
    AUROC table number. The LEFT EDGE is where TPR@1%FPR lives — a curve that hugs the y-axis
    catches positives before its first false alarms. matplotlib so it renders on GitHub."""
    _use_style()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    # Solid lines, full opacity (user decision 2026-07-30; dashed variant tried and rejected).
    # KNOWN LIMIT: near-ceiling curves are COINCIDENT, so the last-drawn probe covers the others
    # on ceiling eval sets — read the per-probe AUROCs from the legend/table there.
    for i, (probe_name, (probe, pooling)) in enumerate(grid.items()):
        sc = probe_scores(probe, pooling, ev["mean"], ev["last"], ev.get("full"), ev.get("fmask"))
        y = ev["labels"].numpy()
        fpr, tpr, _ = roc_curve(y, sc)
        ax.plot(fpr, tpr, color=PROBE_COLORS[i % len(PROBE_COLORS)], lw=1.8,
                label=f"{probe_name}  ({roc_auc_score(y, sc):.3f})")
    ax.plot([0, 1], [0, 1], color="#888", lw=1.2, ls="--", label="Chance", zorder=0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")               # ROC space is a unit square; keep the diagonal a true 45°
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    _finish(fig, save)


def get_pca_components(acts: Tensor, k=2, return_var=False):
    """Top-k PCs of the centered activations. return_var also gives each PC's explained-variance
    fraction (from the singular values), so axes can be labelled with what they explain."""
    X = acts.float() - acts.float().mean(0)
    _, S, V = t.linalg.svd(X, full_matrices=False)
    if return_var:
        return V[:k].T, ((S ** 2) / (S ** 2).sum())[:k]
    return V[:k].T  # [d, k]


def pca_and_projection_panel(mean_acts, labels, layer, direction, title, save=None):
    """Left: unsupervised PCA of the class geometry. Right: projection onto the (supervised) probe
    direction — the axis the probe actually scores on. (Axis + shared-bin fixes: plan #42.)
    matplotlib so it renders on GitHub; the shared explicit bin grid is preserved so bar heights
    stay comparable across the two classes."""
    X = mean_acts[:, layer, :].float()
    Xc = X - X.mean(0)
    pcs, var = get_pca_components(X, k=2, return_var=True)
    proj = (Xc @ pcs).numpy()
    score = (X @ direction).numpy()
    lab = labels.numpy()

    # One shared bin grid for BOTH classes, so bar heights mean the same thing in each.
    lo, hi = float(score.min()), float(score.max())
    pad = 0.02 * (hi - lo) or 1e-6
    edges = np.linspace(lo - pad, hi + pad, 41)   # 40 equal-width bins spanning both classes

    n_pos, n_neg = int((lab == 1).sum()), int((lab == 0).sum())
    _use_style()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.2))

    for cls, color, nm in [(0, NEG_COLOR, "negative"), (1, POS_COLOR, "positive")]:
        axl.scatter(proj[lab == cls, 0], proj[lab == cls, 1], c=color, s=22, alpha=0.7,
                    edgecolors="white", linewidths=0.3, label=nm)
        axr.hist(score[lab == cls], bins=edges, color=color, alpha=0.65, label=nm)

    axl.set_xlabel(f"PC1 ({100 * float(var[0]):.1f}% of variance)")
    axl.set_ylabel(f"PC2 ({100 * float(var[1]):.1f}% of variance)")
    axl.set_title(f"{title}: PCA (centered)", fontsize=12)

    axr.axvline(0.0, color="#333", lw=1.2, ls=":", zorder=5)   # MM threshold anchor, drawn over the bars
    axr.set_xlabel("projection onto probe direction (raw score)")
    axr.set_ylabel(f"count  (n={n_pos} pos / {n_neg} neg)")
    axr.set_title(f"{title}: projection onto probe direction", fontsize=12)

    # A legend under EACH panel (not one shared legend between them).
    for ax in (axl, axr):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, framealpha=0.9)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)     # reserve room for the two under-panel legends
    _finish(fig, save)


def render_token_heatmap_mpl(str_tokens, scores, mask, title="", label="", score_value=None,
                             centering_value=None, score_name="probe score", sequential=False,
                             save=None):
    """Static matplotlib twin of visualize_token_scores — colored token boxes that RENDER ON GITHUB.
    (circuitsvis is a JS widget GitHub can't draw, and inline-CSS spans get sanitized away, so a
    rasterized image is the only fallback that survives.) Applies the SAME mask→center→/std→symmetric
    transform as visualize_token_scores so the two outputs agree, then tiles contiguous colored boxes
    (RdBu_r: red = high score, blue = low, white = centered) wrapped to fixed-width monospace rows.
    Visualization only — not the probe's scoring path."""
    from matplotlib.patches import Rectangle
    if isinstance(scores, Tensor): scores = scores.cpu().tolist()
    if isinstance(mask, Tensor): mask = mask.cpu().tolist()

    # --- identical transform to visualize_token_scores (masked tokens → 0 → white) ---
    unmasked = [s for s, m in zip(scores, mask) if m]
    sc = [s * m for s, m in zip(scores, mask)]
    if centering_value is not None:
        sc = [(s - centering_value) * m for s, m in zip(scores, mask)]
    std = float(np.std(unmasked)) if unmasked else 0.0
    if std > 0:
        sc = [s / std for s in sc]
    vmax = max(max(sc), -min(sc)) if sc else 1.0
    vmax = vmax or 1.0                                  # symmetric range, avoid /0 on an all-white row

    # --- wrap tokens into fixed-width rows (monospace: 1 char ≈ 1 column, so boxes tile cleanly) ---
    toks = process_str_tokens(str_tokens)
    MAXCOL = 100
    rows, cur, w = [], [], 0
    for tok, s, mk in zip(toks, sc, mask):
        score_str = f"{s:+.2f}" if mk else ""
        tl = max(len(tok), len(score_str), 1)   # widen a narrow token (e.g. ",") to fit its score
        if w + tl > MAXCOL and cur:
            rows.append(cur); cur, w = [], 0
        cur.append((tok, s, tl, bool(mk), score_str)); w += tl
    if cur:
        rows.append(cur)

    if sequential:                       # attention weights: ≥0, no centering → green intensity
        cmap = ATTN_CMAP
        norm = plt.Normalize(0.0, vmax)
    else:                                # probe scores: diverging red (high) / blue (low), white at center
        cmap = plt.get_cmap("RdBu_r")
        norm = plt.Normalize(-vmax, vmax)
    _use_style()
    nrows = len(rows)
    fig, ax = plt.subplots(figsize=(11, 0.5 * nrows + 0.9))
    ax.set_xlim(0, MAXCOL); ax.set_ylim(-nrows, 0.0)
    ax.axis("off"); ax.grid(False)
    for ri, row in enumerate(rows):
        x = 0
        for tok, s, tl, mk, score_str in row:
            if mk:
                r, g, b, _ = cmap(norm(s))
                facecolor = (r, g, b)
                txtcol = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "#111"
            else:                                        # unprobed: neutral grey, NOT on the colormap
                facecolor, txtcol = "#ededed", "#999"    # (else viridis-0 would paint them purple)
            ax.add_patch(Rectangle((x, -ri - 1), tl, 1, facecolor=facecolor,
                                   edgecolor="white", linewidth=0.4))
            ax.text(x + tl / 2, -ri - 0.40, tok, family="monospace", fontsize=8,
                    ha="center", va="center", color=txtcol)
            if mk:                                       # score only under the colored (probed) tokens
                ax.text(x + tl / 2, -ri - 0.74, score_str, family="monospace", fontsize=6,
                        ha="center", va="center", color=txtcol)
            x += tl
    head = title
    if score_value is not None:
        head += f"   {score_name} {score_value:.2f}"
    if label:
        head += f"   ground truth label: {label}"
    ax.set_title(head, fontsize=11, loc="left")
    fig.tight_layout()
    _finish(fig, save)


def render_token_heatmap_html(str_tokens, scores, mask, title="", label="", score_value=None,
                              centering_value=None, score_name="probe score", sequential=False,
                              show=True):
    """Interactive HTML token strip using the SAME colormap as render_token_heatmap_mpl (viridis for
    attention, RdBu_r for scores) — circuitsvis can only do a 2-color ramp, so this is how we match
    viridis. Hover a token for its raw value; UNPROBED tokens are greyed rather than painted with the
    colormap (viridis-0 would otherwise show purple). Notebook-only — GitHub sanitizes inline styles."""
    if isinstance(scores, Tensor):
        scores = scores.cpu().tolist()
    if isinstance(mask, Tensor):
        mask = mask.cpu().tolist()

    # same mask→center→/std→symmetric transform as the mpl / circuitsvis paths
    unmasked = [s for s, m in zip(scores, mask) if m]
    sc = [s * m for s, m in zip(scores, mask)]
    if centering_value is not None:
        sc = [(s - centering_value) * m for s, m in zip(scores, mask)]
    std = float(np.std(unmasked)) if unmasked else 0.0
    if std > 0:
        sc = [s / std for s in sc]
    vmax = max(max(sc), -min(sc)) if sc else 1.0
    vmax = vmax or 1.0
    cmap, norm = ((ATTN_CMAP, plt.Normalize(0.0, vmax)) if sequential
                  else (plt.get_cmap("RdBu_r"), plt.Normalize(-vmax, vmax)))

    toks = process_str_tokens(str_tokens)
    spans = []
    for tok, s, raw, m in zip(toks, sc, scores, mask):
        safe = _html.escape(tok)
        if m:
            r, g, b, _ = cmap(norm(s))
            fg = "#fff" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "#111"
            spans.append(f"<span style='background:{to_hex((r, g, b))};color:{fg};padding:1px 2px;"
                         f"border-radius:2px' title='raw {raw:+.4f}'>{safe}</span>")
        else:                                            # unprobed: grey, not on the colormap
            spans.append(f"<span style='color:#999' title='not probed'>{safe}</span>")

    parts = []
    if title:
        parts.append(f"<div style='font:600 14px sans-serif;margin:2px 0'>{_html.escape(str(title))}</div>")
    sub = []
    if score_value is not None:
        sub.append(f"{_html.escape(score_name)}: {score_value:+.3f}")
    if label:
        sub.append(f"ground truth label: <b>{_html.escape(str(label))}</b>")
    if sub:
        parts.append(f"<div style='font:13px sans-serif;color:#333;margin-bottom:2px'>{' · '.join(sub)}</div>")
    stops = ", ".join(to_hex(cmap(x)) for x in np.linspace(0, 1, 16))   # gradient legend matches the map
    parts.append("<div style='display:flex;align-items:center;gap:6px;margin:3px 0;font:11px sans-serif;"
                 f"color:#555'><span>low</span><div style='height:12px;width:220px;"
                 f"background:linear-gradient(to right,{stops});border:1px solid #ccc;border-radius:2px'></div>"
                 "<span>high</span><span style='margin-left:12px'>"
                 "<span style='background:#ededed;border:1px solid #ccc'>&nbsp;&nbsp;</span> unprobed</span></div>")
    parts.append("<div style='font-family:monospace;font-size:13px;line-height:2.0;white-space:pre-wrap'>"
                 + "".join(spans) + "</div>")
    html_str = "".join(parts)
    if show:
        display(HTML(html_str))
    return html_str


def show_token_heatmaps(texts, det_masks, labels, direction, layer, title, pos_name, neg_name,
                        n_show=2, score_name="probe score", save_prefix=None):
    """[NB solutions]-style per-token heatmaps: score all shown examples, compute ONE shared
    centering value from their masked tokens, then render. Colored = detection mask AND not a
    special token (BOS is an attention sink; unmasking it saturates the scale — fix 2026-07-13).
    `score_name` names the quantity in the headers (e.g. "deceptiveness score"); the label shown is
    the dataset's GROUND-TRUTH class, not a prediction. Visualization only — the probe grid scores
    through probe.score, not this path.

    `save_prefix` saves the FIRST shown example of each class as <prefix>_<class-slug>.png (plan
    #59) — deterministic names for the report's paired figure, e.g. heatmap_deceptive/heatmap_honest
    (splits are seed-fixed, so "first of each class" is stable across runs)."""
    model, tokenizer = runtime.get_model(), runtime.get_tokenizer()
    special = set(tokenizer.all_special_ids)
    results, pooled = [], []
    for i in range(n_show):
        str_tokens, scores = score_tokens_with_probe(texts[i], model, tokenizer, direction, layer)
        ids = tokenizer(texts[i], truncation=True, max_length=2048)["input_ids"]
        m = t.zeros(len(scores), dtype=t.bool)
        dm = det_masks[i][: len(scores)].bool()
        m[: len(dm)] = dm
        m &= t.tensor([tid not in special for tid in ids[: len(scores)]])
        results.append((str_tokens, scores, m, i))
        pooled.append(scores[m])
    center = t.cat(pooled).mean().item()
    saved_classes = set()
    for str_tokens, scores, m, i in results:
        lbl = pos_name if labels[i] == 1 else neg_name
        save = None
        if save_prefix and lbl not in saved_classes:
            saved_classes.add(lbl)
            save = f"{save_prefix}_{_slug(lbl)}"
        # Header shows the RAW mean projection (comparable across examples); the per-token boxes are
        # standardized (std from this example's mean) for contrast — name them apart to avoid confusion.
        header_name = f"mean {score_name} (raw)"
        visualize_token_scores(str_tokens, scores, mask=m, score_value=scores[m].mean().item(),
                               centering_value=center, title=f"{title} - example {i}", label=lbl,
                               show=True, score_name=header_name)
        render_token_heatmap_mpl(str_tokens, scores, mask=m, score_value=scores[m].mean().item(),
                                 centering_value=center, title=f"{title} - example {i}", label=lbl,
                                 score_name=header_name, save=save)


def show_attention_heatmaps(full_acts, fmasks, texts, labels, probe_seeds, title, pos_name, neg_name,
                            n_show=2, save_prefix=None):
    """WHERE the attention probe looks ([NB] §5; [HS Fig. 7] analogue): per-token softmax weights,
    averaged over the 3 seeds. Weights are >= 0 and sum to 1 over unmasked positions, so intensity
    = share of attention; no centering. For high-stakes the fmask covers ALL real tokens — mass on
    BOS/template tokens is a finding (attention-sink parking), not a rendering bug.
    `save_prefix`: first shown example per class -> <prefix>_<class-slug>.png (plan #59)."""
    tokenizer = runtime.get_tokenizer()
    saved_classes = set()
    for i in range(n_show):
        with t.no_grad():
            w = t.stack([p.get_attention_weights(full_acts[i:i+1], fmasks[i:i+1])[0, :, 0]
                         for p in probe_seeds]).mean(0)
        ids = tokenizer(texts[i], truncation=True, max_length=full_acts.shape[1])["input_ids"]
        str_tokens = [tokenizer.decode(tid) for tid in ids]
        n = len(str_tokens)
        lbl = pos_name if labels[i] == 1 else neg_name
        save = None
        if save_prefix and lbl not in saved_classes:
            saved_classes.add(lbl)
            save = f"{save_prefix}_{_slug(lbl)}"
        # Interactive: circuitsvis (keeps hover-for-value), highlight in GREEN; plus the GitHub-safe PNG.
        visualize_token_scores(str_tokens, w[:n], mask=fmasks[i][:n], score_value=float(w.max()),
                               title=f"{title} - attention weights, example {i}", label=lbl,
                               show=True, score_name="max attention weight", positive_color="green")
        render_token_heatmap_mpl(str_tokens, w[:n], mask=fmasks[i][:n], score_value=float(w.max()),
                                 title=f"{title} - attention weights, example {i}", label=lbl,
                                 score_name="max attention weight", sequential=True, save=save)


def probe_direction_cosines(grid, mean_acts, last_acts, labels, title, save=None):
    """Cosine similarity between the five probe DIRECTIONS at the detection layer — the MECHANISM
    check for the read/write dissociation ([Apollo App. D.1] LR reads best; [GoT Table 2] MM writes
    best). Directions are normalized explicitly (none is unit-norm) and re-oriented negative→positive
    (as calibrate() does) so signs don't create spurious anti-correlation."""
    def oriented_unit(d, ref_acts):
        d = d.detach().float()
        gap = ref_acts[labels == 1].float().mean(0) - ref_acts[labels == 0].float().mean(0)
        if float(gap @ d) < 0:
            d = -d                       # point negative -> positive, as calibrate() does
        return d / (d.norm() + 1e-8)

    names, dirs = [], []
    for name, (probe, pooling) in grid.items():
        ref = last_acts if pooling == "last" else mean_acts
        d = (t.stack([pr.W_out.data[:, 0] for pr in probe]).mean(0)   # 3-seed mean, as steered
             if pooling == "attention" else probe.direction)
        names.append(name)
        dirs.append(oriented_unit(d, ref))
    D = t.stack(dirs)
    C = (D @ D.T).numpy()

    _use_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    # Scale from the smallest similarity in the matrix up to 1.0, so the colormap spends its full
    # range on the values that occur (the diagonal is 1.0, so C.min() is the smallest off-diagonal).
    vmin, vmax = float(C.min()), 1.0
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    # origin="lower" so the y-axis runs the same direction as x (LR-attention far-right AND top);
    # the self-similarity diagonal then runs bottom-left → top-right.
    im = ax.imshow(C, cmap=cmap, norm=norm, origin="lower")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.grid(False)                                     # the darkgrid style would streak the heatmap
    for i in range(len(names)):
        for j in range(len(names)):
            # Pick text color from the CELL's own luminance, so it stays readable under any colormap
            # (Blues darkens with value; magma/viridis brighten) — no per-colormap threshold to tune.
            r, g, b, _ = cmap(norm(C[i, j]))
            txt = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "#222"
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=9, color=txt)
    ax.set_title(f"{title} — probe direction cosine similarity", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _finish(fig, save)

    # The matrix uses the 3-seed MEAN LR-attention direction, which hides seed disagreement —
    # report it, mirroring the per-seed AUROC print in the grid section.
    sd = [oriented_unit(pr.W_out.data[:, 0], mean_acts) for pr in grid["LR-attention"][0]]
    pairs = [float(sd[i] @ sd[j]) for i in range(len(sd)) for j in range(i + 1, len(sd))]
    print(f"LR-attention pairwise SEED cosines: {[f'{c:.3f}' for c in pairs]}")
    return C


# =========================================================================================
# Steering plots (cells 38, 39)
# =========================================================================================

def plot_steering_effects(results, title):
    """Three panels, one curve per direction (colors pinned via STEER_COLORS):
      1. NIE vs coef            — the causal dose-response (readout shift, class-gap units).
      2. judge BEHAVIOUR vs coef — does generated text exhibit the behaviour more as the dose rises?
      3. judge COHERENCE vs coef — the guard: a behaviour rise that coincides with a coherence
         collapse is "steering by breaking the model", not a causal win. Read 2 and 3 together.
    Panels 2-3 are NaN without ANTHROPIC_API_KEY; panel 1 always fills.

    NO save= HERE, deliberately (plan #59): this is the one remaining plotly figure, and plotly PNG
    export needs kaleido + a headless Chrome (~15 apt libs, wiped on recycle) — rejected 2026-08-05.
    The dose-response panels are manual screenshots -> research-report/figures/<behaviour>/
    steering_<eval>.png (slot names in figures/README.md)."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        f"readout NIE", f"judge: behaviour", f"judge: coherence"))
    for method, r in results.items():
        c = STEER_COLORS.get(method, "#888888")
        fig.add_trace(go.Scatter(x=r["coef"], y=r["nie"], name=method,
                                 line=dict(color=c), marker=dict(color=c)), row=1, col=1)
        for col, key in ((2, "judge_beh"), (3, "judge_coh")):
            if not all(np.isnan(r[key])):
                fig.add_trace(go.Scatter(x=r["coef"], y=r[key], name=method,
                                         line=dict(color=c), marker=dict(color=c),
                                         showlegend=False), row=1, col=col)
    for col in (1, 2, 3):     # every panel gets labelled axes (user request 2026-07-30)
        fig.update_xaxes(title_text="steering coefficient (1.0 = cross mean gap)", row=1, col=col)
    fig.update_yaxes(title_text="NIE (norm. readout shift)", row=1, col=1)
    fig.update_yaxes(title_text="judge: behaviour score (1-10)", range=[1, 10], row=1, col=2)
    fig.update_yaxes(title_text="judge: coherence score (1-10)", range=[1, 10], row=1, col=3)
    fig.update_layout(height=400, width=1250, title=title)
    fig.show()


def show_qualitative(spec, dirs, prompt_text, coef=1.0):
    """Baseline vs EVERY direction, on one prompt, at a single calibrated coef (+1 = one class-mean
    gap) so MM/LR/LR-attention are on equal footing and any difference reflects the METHOD, not the
    dose. Illustrative only; the quantitative claim lives in the grids."""
    base = generate_steered(prompt_text, None)
    rows = [f"<p><b>Baseline (coef 0):</b> {base}</p>"]
    for method in dirs:  # MM, LR, LR-attention
        steered = generate_steered(prompt_text, SteeringHook(dirs[method], coef=coef))
        rows.append(f"<p><b>{method} @ coef {coef:+g}:</b> {steered}</p>")
    display(HTML(f"<h4>{spec.name}: one prompt steered toward the behaviour, all directions @ coef {coef:+g}</h4>"
                 + "".join(rows)))
