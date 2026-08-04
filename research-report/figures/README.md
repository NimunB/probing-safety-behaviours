# Figures for `report_v3.tex` — exactly what to capture, and what to name it

Drop PNGs into this folder using the **exact filenames** below. `report_v3.tex` wraps every
`\includegraphics` in `\IfFileExists`, so the report compiles *now* with a labelled placeholder box
and switches to the real image the moment the file appears. No `.tex` edit needed.

**Capture settings.** Screenshot the figure only (no cell borders, no code). The notebooks set a
retina DPI bump (`viz._use_style()`, plan #50), so a direct screenshot at 100% zoom is already
sharp enough for print. Save as PNG. If a panel is very wide, crop to the axes rather than
downscaling.

**Run order.** Capture these *after* the re-run with the 2026-08-02 fixes (transcript-only readout,
completion-only generation, uncapped high-stakes per-token fit, corrected sycophancy few-shot,
false-claim judge prompts). The layer-selection and PCA panels are unaffected by those fixes and
could be captured from the current run, but the dose-response panels **must** come from the new run
or they will not match the numbers in the text.

---

## STATUS as of 2026-08-04 — 9 of 12 slots filled

| status | file | note |
|---|---|---|
| ✅ | `layers_deception.png` | |
| ✅ | `layers_highstakes.png` | |
| ✅ | `layers_sycophancy.png` | |
| ✅ | `pca_deception.png` | |
| ❌ | `pca_highstakes.png` | **WRONG IMAGE — recapture.** Byte-identical to `layers_highstakes.png` (same SHA-256). Its 1.82 aspect ratio matches the layer plots; the two real PCA panels are 2.71. You saved the layer-selection plot twice. Capture `high_stakes.ipynb` **cell 11**, both panels. |
| ✅ | `pca_sycophancy.png` | |
| ⏳ | `dose_deception.png` | needs the re-run — cell 37, AI Liar (OOD) |
| ⏳ | `dose_highstakes.png` | needs the re-run — cell 33, Anthropic-HH (OOD) |
| ⏳ | `dose_sycophancy.png` | needs the re-run — cell 33, math (OOD) |
| ✅ | `heatmap_deception_deceptive.png` | you supplied a pair instead of one image — **better**, both are now wired side by side in Appendix E, since the point is that the two members differ only in context |
| ✅ | `heatmap_deception_honest.png` | |
| ✅ | `roc_deception.png` | now in Appendix E |
| ✅ | `cosines_deception.png` | now in Appendix E |
| ✅ | `niesweep_deception.png` | now in Appendix E |
| ✅ | `attn_sycophancy.png` | now in Appendix E |

**So: recapture `pca_highstakes.png`, and capture the three `dose_*` panels after the re-run.**
Everything else is in and rendering.

The four "optional" images you supplied were good enough to justify their own **Appendix E:
Supplementary Figures**, which also freed body space — the report body is now ~7.95 pages, inside the
8-page limit.

---

## Reference — the full slot list

### Fig. `layers` — detection-layer selection (§III-C)
Pair-aware 5-fold CV AUROC against layer, MM solid + LR overlay, chosen layer marked.

| filename | notebook | cell | note |
|---|---|---|---|
| `layers_deception.png` | `deception.ipynb` | 9 | chosen layer 12 |
| `layers_highstakes.png` | `high_stakes.ipynb` | 9 | chosen layer 13 |
| `layers_sycophancy.png` | `sycophancy.ipynb` | 9 | chosen layer 19 |

### Fig. `pca` — linear structure (§IV opening)
The two-panel output of `pca_and_projection_panel`: **left** PCA of centred activations (expect no
class separation), **right** histogram of projections onto the trained MM direction (expect clean
bimodal separation). Capture **both panels together** — the contrast is the whole point.

| filename | notebook | cell |
|---|---|---|
| `pca_deception.png` | `deception.ipynb` | 12 |
| `pca_highstakes.png` | `high_stakes.ipynb` | 11 |
| `pca_sycophancy.png` | `sycophancy.ipynb` | 11 |

### Fig. `dose` — steering dose–response (§IV-E)
`plot_steering_effects` on the **OOD** eval set for each behaviour: NIE, judge behaviour and judge
coherence against signed coefficient, three directions overlaid. These cells currently store **no
image output**, so they must be run.

| filename | notebook | cell | eval set |
|---|---|---|---|
| `dose_deception.png` | `deception.ipynb` | 37 | AI Liar (OOD) — *not* the ID panel; ID's NIE is unusable (gap 0.058) |
| `dose_highstakes.png` | `high_stakes.ipynb` | 33 | Anthropic-HH (OOD) |
| `dose_sycophancy.png` | `sycophancy.ipynb` | 33 | math (OOD) — the important one: MM's non-monotone peak and coherence cliff |

### Fig. `heatmap` — what the probe reads (§III-B)
One `show_token_heatmaps` pair from `6a`: the **same** item under both labels if available, so the
reader can see identical tokens coloured differently. Crop to two rows.

| filename | notebook | cell |
|---|---|---|
| `heatmap_deception.png` | `deception.ipynb` | 25 |

---

## Optional — only if space allows

| filename | notebook | cell | what |
|---|---|---|---|
| `roc_deception.png` | `deception.ipynb` | 19 | ROC curves per eval set |
| `cosines_deception.png` | `deception.ipynb` | 20 | cosine matrix between probe directions — nice for "do the arms find the same direction?" |
| `niesweep_deception.png` | `deception.ipynb` | 32 | per-layer NIE sweep, showing the chosen block |
| `attn_sycophancy.png` | `sycophancy.ipynb` | 24 | attention-probe weights; relevant because LR-attention is sycophancy's best steerer |

---

## Cell-number caveat

Indices above are from the notebooks as of 2026-08-02. Adding or removing a cell shifts them. If a
number looks wrong, find the cell by its call instead:

| figure | call to search for |
|---|---|
| layers | `viz.plot_layer_selection` |
| pca | `pca_and_projection_panel` |
| dose | `plot_steering_effects` |
| heatmap | `show_token_heatmaps` |
| roc | `plot_roc_curves` |
| cosines | `probe_direction_cosines` |
| attn | `show_attention_heatmaps` |
