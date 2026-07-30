# safety_probes

Linear probes for safety-relevant behaviours — **deception** and **high-stakes** detection, plus
causal validation by steering. MSc thesis code, extracted from
`safety_probes.ipynb` so it no longer depends on the ARENA course repo.

**Headline hypothesis:** [Apollo App. D.1] found LR is the best *detector* of deception;
[GoT Table 2] found MM directions are the most *causal* (best NIE in 7/8 conditions) despite
similar classification accuracy. Does this read/write dissociation hold across safety
behaviours with different ontologies?

Sources: **[GoT]** Marks & Tegmark, *The Geometry of Truth* · **[Apollo]** Goldowsky-Dill et al.,
*Detecting Strategic Deception Using Linear Probes* · **[HS]** McKenzie et al., *Detecting
High-Stakes Interactions with Activation Probes* · **[CAA]** Rimsky et al., *Steering Llama 2 via
Contrastive Activation Addition* · **[NB]** the ARENA `1.3.1_Linear_Probes` exercises notebook.

---

## Setup on a fresh Vast.ai instance

The base Vast image ships Python 3.11, `uv`, and a default venv at `/venv/main`. **A venv is
enough — conda is not needed** (bitsandbytes/torch install fine from wheels). Steps:

```bash
# 0. Clone the repo (or copy this folder over) and cd into it
cd probing-safety-behaviours

# 1. Make an isolated venv. Either reuse /venv/main, or create a dedicated one:
python -m venv /venv/safety && source /venv/safety/bin/activate
#   (uv works too: uv venv /venv/safety && source /venv/safety/bin/activate)

# 2. Install torch FIRST, from the CUDA index that matches the GPU, THEN the rest.
#    cu128 is what the audited results used; a Blackwell card (RTX 50xx/B200) needs cu128+.
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt

# 3. Make `safety_probes` importable from anywhere (the notebooks live in notebooks/, and a
#    VSCode kernel's cwd is the workspace root — either way `import safety_probes` fails without
#    this). One editable install fixes it for every notebook, the UI, and any cwd:
uv pip install -e .

# 4. circuitsvis installs as 0.0.0 from PyPI's default in some setups — if the heatmap cells
#    error, install a real release explicitly (needed for the notebook heatmaps, NOT the UI):
uv pip install "circuitsvis>=1.43"

# 5. Secrets: create .env next to this README (gitignored). HF_TOKEN is REQUIRED (gated Llama
#    repo → 401 without it); ANTHROPIC_API_KEY is optional (judge cells skip cleanly if absent).
printf 'HF_TOKEN=hf_xxx\nANTHROPIC_API_KEY=sk-ant-xxx\n' > .env

# 6. Sanity check — imports clean and finds the .env, without loading the model:
python -c "import safety_probes; print('ok')"
```

**Is `requirements.txt` sufficient?** Yes for both the UI and the parity run — it now includes
`fastapi`/`uvicorn` (UI) and `circuitsvis` (heatmaps). Two things it deliberately does *not* pin:
the model weights (re-downloaded from HF on first `runtime.init()`, ~15 GB, needs `HF_TOKEN`), and
`scipy` (only the optional judge-validation step). Provision the instance with **≥120 GB disk** and
**a volume** so the weights + caches survive a recycle.

**Do NOT `pip install gradio`** — it force-downgrades `starlette`/`pydantic` or breaks
`huggingface_hub` for `transformers`. The UI is FastAPI for exactly this reason (plan M9).

---

## Running the UI

Interactive detection + steering over the exported probe bundles. Loads the model once; computes
everything live from your input (no activations stored).

**1. Free the GPU.** The 8-bit 8B needs ~19 GB of a 24 GB card. If a notebook kernel still holds
the model, the UI will OOM — shut the kernel down first, then confirm:

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader     # want >= 20000 MiB
```

**2. Start the server** (binds `127.0.0.1:17860`; adds no network exposure by itself):

```bash
cd probing-safety-behaviours
/venv/arena-env/bin/python ui_probe.py
```

Expect `env: .../.env`, then a warning if `HF_HOME` points at the wrong cache (it auto-corrects —
see below), then a quiet minute or two while Llama loads, then `Serving on http://127.0.0.1:17860`.

**3. Reach it.** It's on localhost, so pick one:

*SSH tunnel (simplest; nothing public, no token):* from your own machine,
```bash
ssh -p $VAST_TCP_PORT_22 -L 8080:127.0.0.1:17860 root@$PUBLIC_IPADDR
# then open http://localhost:8080
```
On Vast, `$VAST_TCP_PORT_22` and `$PUBLIC_IPADDR` are set in the instance shell — read them there
(this box: port 41554, IP 77.48.24.250) and substitute the real values into the command.

*Caddy (shareable URL + token auth), on a reserved free port:*
```bash
python -c "import yaml; d=yaml.safe_load(open('/etc/portal.yaml')) or {'applications':{}}; \
d['applications']['Safety Probes']={'hostname':'localhost','external_port':10100, \
'internal_port':17860,'open_path':'/','name':'Safety Probes'}; \
yaml.safe_dump(d, open('/etc/portal.yaml','w'), sort_keys=False)"
supervisorctl restart caddy
# reachable at $PUBLIC_IPADDR:$VAST_TCP_PORT_10100 with $OPEN_BUTTON_TOKEN
```

### Two Vast gotchas the code now handles for you

* **`.env`** — `runtime.init()` searches `$SAFETY_PROBES_ENV` → `<repo>/.env` → each parent dir, so
  it finds `exercises/.env` today and a repo-local `.env` once standalone. Missing `HF_TOKEN` fails
  loudly (gated Llama repo → 401), not with a wall of traceback.
* **`HF_HOME`** — Vast sets `HF_HOME=${WORKSPACE}/.hf_home`, but the weights may live at
  `/root/.cache/huggingface`. `runtime.init()` picks whichever cache actually holds the model and
  warns if that isn't `$HF_HOME`, so it will **not** silently re-download 15 GB.

---

## Extraction status — COMPLETE, pending the parity run

Every function/class in `safety_probes.ipynb` is now either in the `safety_probes` package or a
driver-local helper in a notebook. Full-package AST audit: **56/56 shared definitions faithful** to
the notebook (49 byte-identical after mechanical `runtime.` substitutions; 7 cosmetic — type
annotations, context-manager methods, import placement). The three `notebooks/` orchestrate the
pipeline by importing the package.

**What's left is the parity RUN** — execute `notebooks/deception.ipynb` and `high_stakes.ipynb` on
the GPU and confirm the numbers match the audited notebook: detection layer **12 / 20**, the grid
AUROCs, steering block **[15-19] / [9-13]**, the NIE curves. Until that passes,
`safety_probes.ipynb` remains the source of truth. A refactor that moves a number is a bug.
(`summary.ipynb` was dissolved 2026-07-30, plan #45 — each notebook now computes its own probing
verdict at the end of its probing section and steering verdict at the end of its steering section.)

| Module | Status | Contents |
|---|---|---|
| `runtime.py` | ✅ | model/tokenizer/device/config + env & HF-cache resolution; replaces notebook globals |
| `chat.py` | ✅ | **the ARENA cut** — vendored `build_detection_mask`, `clean_bpe_token`, + the 3 chat helpers |
| `probes.py` | ✅ | `MMProbe`, `LRProbe`, `AttentionProbe`, `train_attention_probe` |
| `extraction.py` | ✅ | cache + pooled/full-sequence extractors |
| `spec.py` | ✅ written | `BehaviourSpec` |
| `export.py` | ✅ written | `export_behaviour`, `load_behaviour`, `rebuild_detection_probe` |
| `metrics.py` | ✅ written, unit-tested | grid training, `probe_scores`, `evaluate_grid`, `bootstrap_auroc`, `tpr_at_fpr`, `cv_auroc_by_layer` |
| `datasets.py` | ✅ written, helpers unit-tested | deception pairs, AI Liar, Roleplaying, MUP high-stakes + OOD loaders |
| `steering.py` | ✅ AST-verified | full WRITE side: `SteeringHook`, `calibrate`, `generate_steered`, `compose_steering`, `readout_pdiff`, `nie_baseline_gap`, `nie_layer_sweep`, `pick_block`, `build_directions`, `calibrate_block` |
| `judge.py` | ✅ AST-verified | Claude judge (`enum` fix #44), lazy client |
| `evaluation.py` | ✅ AST-verified | `steering_nie_llm_evaluation` (steering × judge) |
| `viz.py` | ✅ AST-verified | heatmaps, PCA panel, ROC, steering plots, cosine matrix + vendored ARENA viz utils |
| `summary.py` | ✅ | verdict helpers `auroc_pivot` / `nie_at` + CSV persistence of final numbers (verdict cells live in the behaviour notebooks, plan #45) |
| `notebooks/` | ✅ built | `deception.ipynb` (§2), `high_stakes.ipynb` (§3) — per-behaviour verdicts inline; `summary.ipynb` removed 2026-07-30; `comparison.ipynb` (plan #48) — cross-behaviour steering overlays (GPU-free, from `exported_probes/*_steering_full.json`) + drafted simultaneous-steering experiment |

`ui_probe.py` now imports the loaders from `safety_probes.export` (the duplication is gone).

### Running the parity check

```bash
# free the GPU first (see § Running the UI), then run in order:
jupyter nbconvert --to notebook --execute notebooks/deception.ipynb --output deception.ipynb
jupyter nbconvert --to notebook --execute notebooks/high_stakes.ipynb --output high_stakes.ipynb
```

Or run them interactively. Compare against the audited `safety_probes.ipynb`: **detection layer
12 / 20**, the grid AUROC tables, **steering block [15-19] / [9-13]**, the NIE dose-response. The two
EDIT-HERE points default to `argmax` / `pick_block`; pin them to the audited values to reproduce the
exported bundles byte-for-byte. `circuitsvis` must be installed for the heatmap cells (see
requirements.txt).

---

## The one design decision worth knowing

The notebook keeps `model`, `tokenizer`, `SEED`, `MODEL_NAME` as globals; modules can't. Rather
than threading `model`/`tokenizer` through every signature — which would change nearly every
function and make a diff against the audited notebook unreadable — shared state lives in
`runtime.py` and other modules read `runtime.model` / `runtime.tokenizer`.

**Consequence: `runtime.init()` must be called once before anything touches the model.** The
accessors raise a clear error rather than an `AttributeError` if you forget.

```python
from safety_probes import runtime, chat
runtime.init()                       # loads Llama-3.1-8B-Instruct in 8-bit
text, mask = chat.chat_with_assistant_mask(messages)
```

## Why `chat.py` is load-bearing

Every activation the probes see is indexed by a mask built in `chat.py`. If templating and
masking disagree by one token, every probe score is computed on the wrong activations —
silently. That was the double-BOS bug (plan #22). **Any consumer, including the UI, must
tokenize through these helpers** rather than calling `apply_chat_template` directly.

`build_detection_mask` is vendored **verbatim**, double-BOS quirk included, with the strip-fix
left in the caller where the notebook has it. We now own the code and *could* fix it at source,
but that would invalidate the line-by-line audit and every cached activation. Revisit only with
a cache wipe and a full re-run.

## Data is vendored and pinned

`data/` holds the three Apollo files (500 KB) frozen at commit `f8ec4010`, with SHA-256 digests in
[`data/PROVENANCE.md`](data/PROVENANCE.md). The upstream repo is 535 MB for those same three files
and was **unpinned** — an upstream edit would have silently changed the thesis's data, and the
content-blind cache would not have recomputed.

`Arrrlex/models-under-pressure` stays a HuggingFace fetch but is pinned via `MUP_REVISION`.

`data/` and `exported_probes/` are deliberately **not** gitignored; `activation_cache/` and the
fallback clone are.

## Cache caveat (carried over deliberately)

The activation-cache key is `(kind, model, name, n_examples, max_length)` — **content-blind**.
Edit a loader without changing `name` or the example count and the cache silently serves stale
activations. After editing any loader, delete the matching `activation_cache/*.pt`.

---

## Remaining Phase 1 work

1. Extract `datasets.py`, `metrics.py`, `steering.py`, `judge.py`, `viz.py`.
2. Build `notebooks/deception.ipynb` and `notebooks/high_stakes.ipynb` importing from the package.
3. **Parity run** — execute both notebooks here and confirm the numbers match the audited
   notebook. Same detection layer, same AUROCs, same steering block. This is the gate; a
   refactor that changes a number is a bug, not a refactor.
4. Push as its own repo, then migrate to a fresh instance with a volume.

## Known gap: figures don't render on GitHub

Plotly output is stored as `application/vnd.plotly.v1+json` (GitHub can't render it) and the
heatmaps are inline-styled HTML (GitHub strips `style` attributes, so they lose all colour).
There is no `image/png` anywhere in the notebook as a fallback. Options: view via **nbviewer**
(zero code change), or install `kaleido` and emit a PNG alongside each figure. Worth doing when
the plotting call sites move into `viz.py` rather than retrofitting later.

## Environment

Developed on a Vast.ai RTX 3090 (24 GB), Llama-3.1-8B-Instruct in 8-bit, `/venv/arena-env`.
See `requirements.txt` — install torch from the CUDA index first.
