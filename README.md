# safety_probes

Linear probes for safety-relevant behaviours in Llama-3.1-8B-Instruct, and calibrated activation
steering along the same directions. MSc thesis code.

The question the repo is built around: **when you find a linear direction for a safety behaviour, is
the direction that best *detects* the behaviour also the one that best *steers* it?** Prior work
answers those two halves separately and on different properties. [Apollo App. D.1] finds logistic
regression the best *detector* of deception; [GoT Table 2] finds difference-of-means directions the
most *causal* (best NIE in 7 of 8 conditions) despite comparable classification accuracy. Read
together that implies a **read/write dissociation**, but nobody tests it in one pipeline, so the
apparent effect could just be an artefact of differing models, metrics and token policies. This repo
holds everything else fixed and measures both sides on the same activations.

Three behaviours are implemented: **deception**, **high-stakes awareness**, and **sycophancy**. The
answer turns out to depend on which: the dissociation replicates on the first two and inverts on the
third.

Sources: **[GoT]** Marks & Tegmark, *The Geometry of Truth* · **[Apollo]** Goldowsky-Dill et al.,
*Detecting Strategic Deception Using Linear Probes* · **[HS]** McKenzie et al., *Detecting
High-Stakes Interactions with Activation Probes* · **[CAA]** Rimsky et al., *Steering Llama 2 via
Contrastive Activation Addition* · **[NB]** the ARENA `1.3.1_Linear_Probes` exercises notebook.

---

## What's in here

```
safety_probes/          the pipeline as an importable package (~2,900 lines)
  behaviours.py           the three BehaviourSpecs — the only behaviour-specific code
  spec.py                 the BehaviourSpec dataclass
  runtime.py              model/tokenizer/device/config; replaces notebook globals
  chat.py                 chat templating + detection masks (see "load-bearing", below)
  datasets.py             loaders for all three behaviours and their OOD sets
  extraction.py           activation extraction + the disk cache
  probes.py               MMProbe, LRProbe, AttentionProbe
  metrics.py              the probe grid, AUROC/TPR, pair-aware CV layer sweep
  steering.py             the whole write side: hooks, calibration, NIE readout, block selection
  judge.py                the LLM judge (behaviour + coherence)
  evaluation.py           couples steering to the judge
  viz.py, export.py, summary.py

notebooks/              one driver notebook per behaviour
ui_probe.py             interactive scoring + steering over the exported bundles
data/                   vendored and pinned inputs
exported_probes/        frozen probes, directions and result tables from the last run
research-report/        the thesis report (LaTeX), its figures, and a table generator
```

Adding a behaviour means writing one `BehaviourSpec` and one loader. Nothing downstream is
conditioned on which behaviour is running, which is what makes the three comparable.

## How the pipeline works

Each behaviour flows through the same stages, with the read and write sides diverging after
extraction:

1. **Mask.** Decide which tokens the probe may see. The rule is *mask where the label lives*:
   assistant content only for deception and sycophancy (their system prompts state or imply the
   label), every role's message content for high-stakes (the danger can be stated by anyone).
2. **Extract.** One forward pass per batch keeps all 32 layers, pooled two ways (`mean` and `last`)
   and cached as fp16. This is what makes a full layer sweep cheap.
3. **Select a layer, twice.** Detection layer by pair-aware 5-fold cross-validated AUROC on the
   training split. Steering block by a causal sweep: compute the NIE at every layer, take the
   contiguous five-layer window with the largest summed effect. The two rarely coincide.
4. **Probe grid.** Five cells crossing estimator (mass-mean, logistic regression) with pooling
   (last, mean, learned attention), so a difference can be attributed to one axis rather than both.
5. **Steer.** Refit directions per block layer, calibrate so coefficient 1.0 crosses the class-mean
   gap, then measure two ways: the **NIE** on a constrained YES/NO readout, and an **LLM judge**
   scoring generated text for the behaviour *and*, separately, for coherence. The coherence axis
   exists because a readout metric cannot tell "induced the behaviour" from "broke the model".

## Results at a glance

Best out-of-distribution detector, and best steerer, per behaviour:

| | Deception | High-stakes | Sycophancy |
|---|---|---|---|
| Detection layer / steering block | 12 / 11–15 | 20 / 6–10 | 19 / 6–10 |
| Best OOD detector | LR (mean) 0.722 | LR (mean) 0.774 | **MM (last) 0.879** |
| Best steerer (NIE) | MM | MM | **LR-attention** |
| Read/write dissociation | holds | holds | **inverts** |

Full tables are in `exported_probes/*.csv` and in the report's appendices, which are generated from
those files by `research-report/make_appendix_tables.py`.

## The interactive interface

`ui_probe.py` serves the exported probes and directions over a small local web app: paste text and
score it with any probe in the grid, or steer generation with any direction at a chosen coefficient.
It computes everything live from your input and stores no activations. It shares `chat.py` and
`export.py` with the pipeline, so what it scores is what the notebooks scored.

---

## Setup

**Requirements.** An NVIDIA GPU with at least 24 GB of memory (RTX 3090/4090/5090, A5000, or
similar) — this applies to the notebooks and the UI alike, since both load the live model. The
model loads in 8-bit via bitsandbytes, which is CUDA-only, so Apple Silicon and CPU-only machines
cannot run either; from such a machine, reach a GPU box over the SSH tunnel described below.
Expect ~15 GB of disk for the model download plus several GB for the activation cache. The
LLM-judge cells need an `ANTHROPIC_API_KEY` and skip cleanly without one.

```bash
# 1. Isolated venv (or reuse /venv/main).
python -m venv /venv/safety && source /venv/safety/bin/activate

# 2. Install torch FIRST from the CUDA index matching the GPU, THEN the rest.
#    A Blackwell card (RTX 50xx / B200) needs cu128 or newer.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 3. Make the package importable from any cwd (notebooks live in notebooks/,
#    but a VSCode kernel's cwd is the workspace root).
pip install -e .

# 4. circuitsvis sometimes resolves to 0.0.0; if the notebook heatmap cells error,
#    pin a real release. Not needed for the UI.
pip install 'circuitsvis>=1.43'

# 5. Secrets in .env next to this README (gitignored).
#    HF_TOKEN is REQUIRED (gated Llama repo, 401 without it).
#    ANTHROPIC_API_KEY is optional; judge cells skip cleanly and leave NaN columns.

# 6. Sanity check, without loading the model:
python -c "from safety_probes import runtime; runtime.load_env(); print('ok')"
```

## Running it

**The behaviour notebooks.** Run `notebooks/deception.ipynb`, `high_stakes.ipynb` and
`sycophancy.ipynb` top to bottom; each loads the model once and ends by exporting its bundle. Two
cells per notebook are marked `EDIT HERE`: the detection layer and the steering block both default to
`argmax` / `pick_block`, and pinning them to the recorded values reproduces the exported bundles
exactly.

**The UI.** Free the GPU first: the 8-bit 8B needs roughly 19 GB, so shut down any notebook kernel
still holding the model, then

```bash
source /venv/main/bin/activate && python ui_probe.py     # binds 127.0.0.1:17860
```

Expect a quiet minute or two while Llama loads. It binds to localhost and adds no network exposure
by itself; reach it with an SSH tunnel

```bash
ssh -p <ssh-port> -L 8080:127.0.0.1:17860 root@<host>   # then open localhost:8080
```

or, on a Vast.ai box, by adding an entry to `/etc/portal.yaml` on a free reserved port and
restarting Caddy, which puts it behind token auth. Read the port and IP from the instance shell;
they change with every instance.

**The report.** `cd research-report && latexmk -pdf report_v3.tex`. Appendices A and B are generated:
re-run `make_appendix_tables.py` after any pipeline run so the tables cannot drift from
`exported_probes/`.

---

## Design notes worth knowing

**`runtime.init()` must be called before anything touches the model.** The notebook kept `model`,
`tokenizer`, `SEED` and `MODEL_NAME` as globals and modules cannot. Rather than thread
`model`/`tokenizer` through every signature, shared state lives in `runtime.py` and other modules
read `runtime.model` / `runtime.tokenizer`. The accessors raise a clear error rather than an
`AttributeError` if you forget.

```python
from safety_probes import runtime, chat
runtime.init()                       # loads Llama-3.1-8B-Instruct in 8-bit
text, mask = chat.chat_with_assistant_mask(messages)
```

**`chat.py` is load-bearing.** Every activation a probe sees is indexed by a mask built there. If
templating and masking disagree by one token, every probe score is computed on the wrong activations,
silently — that was the double-BOS defect. Any consumer, the UI included, must tokenize through these
helpers rather than calling `apply_chat_template` directly. `build_detection_mask` is vendored
verbatim from [NB], double-BOS quirk included, with the strip-fix in the caller; fixing it at source
would invalidate the line-by-line audit and every cached activation, so revisit only with a cache
wipe and a full re-run.

**The activation cache is content-blind.** Its key is `(kind, model, name, n_examples, max_length)`,
so editing a loader without changing `name` or the example count silently serves stale activations.
Delete the matching `activation_cache/*.pt` after any loader change.

**Data is vendored and pinned.** `data/` holds the Apollo files (500 KB) frozen at a recorded commit,
with SHA-256 digests in [`data/PROVENANCE.md`](data/PROVENANCE.md). Upstream is 535 MB for the same
files and was unpinned, so an upstream edit would have silently changed the thesis's inputs, and the
content-blind cache would not have noticed. `Arrrlex/models-under-pressure` stays a HuggingFace fetch
pinned by `MUP_REVISION`. `data/` and `exported_probes/` are deliberately not gitignored;
`activation_cache/` is.

**Environment quirks handled for you.** `runtime.init()` searches `$SAFETY_PROBES_ENV` →
`<repo>/.env` → parent directories for the env file, and fails loudly on a missing `HF_TOKEN` rather
than with a 401 traceback. It also picks whichever HuggingFace cache actually holds the weights and
warns if that isn't `$HF_HOME`, so it will not silently re-download 15 GB.

## Documentation

The full design rationale is written up in the report under `research-report/`; the notebooks
carry their own section-by-section commentary.
