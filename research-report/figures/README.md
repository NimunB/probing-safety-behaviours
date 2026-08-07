# Figures for the report (`report_v4.tex`; `report_v3.tex` predecessor) — auto-saved per behaviour (plan #59)

Since 2026-08-05 the notebooks **save their own report figures**: each driver notebook calls
`viz.save_figures_to("<behaviour>")` in its imports cell, and every `save=`-tagged plot call then
writes `figures/<behaviour>/<slot>.png` as a side effect of rendering. **Every full notebook run
refreshes its behaviour's folder** — no screenshotting, no stale panels, except the one manual case
below. `report_v3.tex` wraps every `\includegraphics` in `\IfFileExists`, so a missing slot renders
as a labelled placeholder box and fills in the moment the file appears.

```
figures/
  deception/     <- deception.ipynb
  high_stakes/   <- high_stakes.ipynb
  sycophancy/    <- sycophancy.ipynb
  ui/            <- manual screenshots (not auto-saved)
```

## Manual slots: `ui/` screenshots

Manual captures, not notebook side effects. `report_v4.tex` references:

- `ui/probing/sycophancy.png` — detection tab, live per-token scores (main text, Fig. 4)
- `ui/high-stakes.png` — single-direction steering demo (appendix)
- `ui/multi-steering/deception-sycophancy.png` — two directions composed (appendix)
- `ui/multi-steering/deception-high-stakes.png` — composition failure at high coefficients (appendix)

`ui/steering/EXTENSION-DIF TYPES.png` is not referenced (it is a ChatGPT screenshot, not the UI).

## The one manual exception: `steering_*.png` (dose-response panels)

`plot_steering_effects` is the repo's one remaining **plotly** figure; PNG export would need
kaleido + a headless Chrome (~15 apt libs, wiped on instance recycle) — rejected 2026-08-05.
**Screenshot these panels yourself** (figure only, 100% zoom, crop to the axes) and drop them into
the behaviour's folder under these exact names:

| file | notebook cell (search for `plot_steering_effects`) |
|---|---|
| `deception/steering_id.png` | `"Deception steering (ID)"` |
| `deception/steering_ai_liar.png` | `"Deception steering (OOD: AI Liar)"` |
| `deception/steering_roleplaying.png` | `"Deception steering (OOD: Roleplaying)"` |
| `high_stakes/steering_id.png` | `"High-stakes steering (ID)"` |
| `high_stakes/steering_ood.png` | `"High-stakes steering (OOD: Anthropic HH)"` |
| `sycophancy/steering_id.png` | `"Sycophancy steering (ID)"` |
| `sycophancy/steering_ood.png` | `"Sycophancy steering (OOD: math)"` |

⚠ The steering screenshots currently in the folders are from the **pre-2026-08-02-fixes run**
(all-token steering scope, old sycophancy readout). Recapture all seven after the re-run —
the dose-response numbers in the text will not match them otherwise.

## Auto-saved slots (refreshed by every run)

Wired into `report_v3.tex` (`<b>` = each behaviour folder):

| slot | source call | note |
|---|---|---|
| `<b>/layers.png` | `viz.plot_layer_selection(..., save="layers")` | CV AUROC by layer, chosen layer marked |
| `<b>/pca.png` | `pca_and_projection_panel(..., save="pca")` | both panels in one image |
| `<b>/cosines.png` | `probe_direction_cosines(..., save="cosines")` | direction cosine matrix |
| `<b>/niesweep.png` | `viz.plot_nie_bars(..., save="niesweep")` | per-layer NIE, chosen block highlighted |
| `deception/roc_ai_liar.png` | `plot_roc_curves(..., save="roc_ai_liar")` | the tex's OOD ROC row uses AI Liar for deception |
| `high_stakes/roc_ood.png`, `sycophancy/roc_ood.png` | `plot_roc_curves(..., save="roc_ood")` | Anthropic-HH / math |
| `deception/heatmap_deceptive.png` + `heatmap_honest.png` | `show_token_heatmaps(..., save_prefix="heatmap")` | first shown example of each class — with seed-fixed splits that's the same pair every run |
| `sycophancy/attn_sycophantic.png` | `show_attention_heatmaps(..., save_prefix="attn")` | first sycophantic attention example |

Also saved (not currently in the tex, available for the appendix): `roc_id`, the OOD heatmap/attn
variants (`heatmap_ood_*`, `attn_ood_*`, deception's `*_ai_liar`/`*_roleplaying`), and the
class-suffixed pairs for the other behaviours (`heatmap_high_stakes`/`heatmap_low_stakes`,
`heatmap_sycophantic`/`heatmap_candid`, `attn_deceptive`/`attn_honest`, …). Class suffixes are
slugged class names (`viz._slug`: lowercase, non-alphanumerics → `_`).

## Conventions

- PNGs at `savefig.dpi=200`, `bbox_inches="tight"` — print-sharp, no zoom fiddling.
- A `save=` tag with no `save_figures_to()` configured prints a loud skip warning, never saves
  silently elsewhere; untagged calls (e.g. from `ui_probe.py`) never save.
- Adding a new slot: tag the plot call with `save="<name>"` in the notebook, reference
  `figures/<behaviour>/<name>.png` in the tex. Filenames are the contract between the two.
