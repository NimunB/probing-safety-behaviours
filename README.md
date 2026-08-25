# Probing and Steering for Safety-Relevant Behaviours in LLMs

MSc project code. Activation probes are computed directions in a language model's residual stream,
used both to **detect** safety-relevant behaviours and, by adding the direction back into the
stream, to **steer** them. This repository crosses two estimators (difference-of-means, logistic
regression) with three token-pooling schemes (last, mean, learned attention) over three behaviours
(deception, high-stakes awareness, sycophancy) on Llama-3.1-8B-Instruct, and evaluates every
direction twice: as a detector, and as a steering vector.

Full method and results: **`Nimun_Bajwa_PG_Project_Dissertation.pdf`**, submitted alongside this
repository.

---

## 1. What is being submitted

| Path | What it is |
|---|---|
| `safety_probes/` | The pipeline as an importable package (16 modules, ~3,100 lines) |
| `notebooks/` | One driver notebook per behaviour — these produce every reported result |
| `ui_probe.py` | Interactive scoring and steering interface over the exported probes |
| `exported_probes/` | Frozen probes, calibrated directions and result tables from the final run |
| `data/` | Vendored and pinned input datasets, with SHA-256 digests in `data/PROVENANCE.md` |
| `requirements.txt` | Package versions pinned to the environment the results were produced in |
| `.env.example` | Template for the two API keys (see step 3 of Setup) |
| `pyproject.toml` | Minimal packaging so `pip install -e .` makes the package importable |

Inside `safety_probes/`:

```
behaviours.py   the three BehaviourSpecs — the only behaviour-specific code
spec.py         the BehaviourSpec dataclass
runtime.py      model/tokenizer/device/config; must be initialised before anything else
chat.py         chat templating + detection masks — every probe score is indexed by these
datasets.py     loaders for all three behaviours and their out-of-distribution sets
extraction.py   activation extraction + the disk cache
probes.py       MMProbe, LRProbe, AttentionProbe
metrics.py      the probe grid, AUROC/TPR, pair-aware cross-validated layer sweep
steering.py     the write side: hooks, calibration, NIE readout, block selection
judge.py        the LLM judge (behaviour + coherence)
evaluation.py   couples steering to the judge
viz.py          figures and heatmaps
export.py       the ~0.5 MB probe bundles
summary.py      verdict helpers
```

## 2. Why there is no executable file, and what to run instead

A standalone executable is not possible for this software, for three reasons:

1. **It requires an NVIDIA GPU.** The model is loaded in 8-bit via `bitsandbytes`, which is
   CUDA-only. At least 24 GB of VRAM is needed (the 8-bit 8B model occupies roughly 19 GB). CPU-only
   and Apple Silicon machines cannot run it at all.
2. **The model weights cannot be bundled.** `meta-llama/Llama-3.1-8B-Instruct` is a gated
   repository: the licence must be accepted per Hugging Face account, so the examiner must supply
   their own `HF_TOKEN`. The weights are also ~15 GB, downloaded on first run.
3. **The software is a research pipeline, not an application.** Its outputs are the measurements in
   the dissertation, produced by running notebooks end to end.

The closest thing to an executable is the interface in section 5: after setup, one command serves a
web app where text can be scored and steered live, using the pre-computed probes in
`exported_probes/`. That path does **not** require re-running the pipeline.

## 3. Setup

Needed once, for both the notebooks and the interface. Allow ~15 GB of disk for the model plus
several GB for the activation cache.

```bash
# 1. Isolated virtual environment.
python -m venv .venv && source .venv/bin/activate

# 2. Install torch FIRST from the CUDA index matching the GPU, THEN the rest.
#    A Blackwell card (RTX 50-series, B200) requires cu128 or newer.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 3. Make the package importable from any working directory.
pip install -e .

# 4. Create a .env file next to this README, using .env.example as the template.
#    HF_TOKEN           REQUIRED — the Llama repo is gated and returns 401 without it.
#    ANTHROPIC_API_KEY  optional — the LLM-judge cells skip cleanly without it, leaving
#                       the judge columns empty; every other number still computes.

# 5. Check the environment without loading the model.
python -c "from safety_probes import runtime; runtime.load_env(); print('ok')"
```

## 4. Reproducing the results

Run the three notebooks top to bottom, in any order. Each loads the model once, runs its behaviour
through the whole pipeline, and ends by writing its bundle to `exported_probes/`.

```
notebooks/deception.ipynb
notebooks/high_stakes.ipynb
notebooks/sycophancy.ipynb
```

Expect a few hours per notebook on a 24 GB card, dominated by the layer sweeps.

Each notebook selects its own detection layer and steering block, so a clean run should arrive at
the values recorded in the dissertation:

| | Detection layer | Steering block |
|---|---|---|
| Deception | 12 | 11–15 |
| High-stakes | 20 | 6–10 |
| Sycophancy | 19 | 6–10 |

**Why it reproduces.** Seed 42 is set at model load, and each stochastic step (splits, bootstrap
resamples, the logistic-regression solver, attention-probe initialisation) seeds its own local
generator, so re-running or re-ordering cells does not change the result. Cross-validation groups
are sorted-index integers rather than Python `hash()`, so the folds are identical across processes.
Inputs are vendored at fixed upstream commits or pinned by Hugging Face revision, decoding is greedy
throughout, and package versions are pinned in `requirements.txt`.

**The exceptions** are the LLM-judge scores, which can shift slightly between runs because Anthropic
does not guarantee identical outputs even at temperature 0, and final digits on different GPU
hardware, where floating-point summation order varies. NIE, the deterministic metric, is what
selected the steering blocks. Appendix S of the dissertation states this in full.

## 5. Running the interface

`ui_probe.py` serves the exported probes and directions over a small local web app: paste a
conversation and score every token with any probe in the grid, or generate two responses to the same
prompt — unsteered and steered — with any direction at a chosen coefficient. It computes everything
live from the input and stores no activations.

Free the GPU first: shut down any notebook kernel still holding the model, then

```bash
source .venv/bin/activate && python ui_probe.py     # binds 127.0.0.1:17860
```

Llama takes a minute or two to load. The app binds to localhost and adds no network exposure by
itself; reach it from another machine with an SSH tunnel:

```bash
ssh -p <ssh-port> -L 8080:127.0.0.1:17860 <user>@<host>   # then open localhost:8080
```

Because it shares `chat.py` and `export.py` with the pipeline, what the interface scores is what the
notebooks scored.
