"""Process-wide runtime state: the model, tokenizer, device, and derived config.

WHY THIS MODULE EXISTS
----------------------
In `safety_probes.ipynb` everything is a notebook global — `model`, `tokenizer`, `device`,
`SEED`, `MODEL_NAME`, `N_LAYERS`, `D_MODEL`, `section_dir`. Functions reference them freely
because a notebook has exactly one namespace. Modules do not.

Two ways to fix that:

  (a) Thread `model` / `tokenizer` through every function signature. Cleanest in the abstract,
      but it changes the signature of nearly every function in the pipeline — which makes a
      line-by-line diff against the audited notebook much harder to read.

  (b) Keep the signatures identical and hold shared state in one module that others read.

We take (b) deliberately: this codebase has been line-by-line audited against the notebook and
the papers, and preserving signatures keeps that audit valid. The cost is that `init()` must be
called before anything else touches the model — the accessors raise a clear error if it isn't.

USAGE
-----
    from safety_probes import runtime
    runtime.init()                      # loads Llama-3.1-8B-Instruct in 8-bit, once
    print(runtime.N_LAYERS, runtime.D_MODEL)

Everything downstream then refers to `runtime.model` / `runtime.tokenizer` rather than a
notebook global. `init()` is idempotent — calling it twice will not reload the weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch as t

# --- Configuration constants (were notebook globals in §1a) ------------------------------
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 42

# --- Populated by init(); None until then -------------------------------------------------
model = None
tokenizer = None
device: t.device = t.device("cuda" if t.cuda.is_available() else "cpu")
N_LAYERS: int | None = None
D_MODEL: int | None = None

# Where the activation cache and exported probes live. Defaults to this package's parent
# (the repo root), NOT the ARENA section directory — set explicitly via init(project_root=...)
# if you want them somewhere else.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _require(name: str):
    """Raise a useful error instead of an AttributeError on None."""
    raise RuntimeError(
        f"safety_probes.runtime.{name} is not initialised. "
        f"Call runtime.init() once before using the pipeline."
    )


def load_env(explicit: Path | None = None) -> Path | None:
    """Load the repo-local .env carrying HF_TOKEN (gated Llama repo) and ANTHROPIC_API_KEY (judge).

    Standalone by design: looks ONLY at <project_root>/.env — i.e. probing-safety-behaviours/.env,
    the directory this package lives in — and never at parent directories, so the repo carries its
    own credentials and does not depend on any outside layout (e.g. the old ARENA exercises/.env).
    Pass `explicit` to point at a specific file instead.

    Returns the path loaded, or None. Without HF_TOKEN, `from_pretrained` fails with a
    GatedRepoError 401 — an auth failure, not a network or model-name problem.
    """
    from dotenv import load_dotenv

    path = Path(explicit) if explicit else PROJECT_ROOT / ".env"
    if path.is_file():
        load_dotenv(dotenv_path=str(path), override=False)
        return path
    return None


def _model_cache_size(hf_home: Path, model_name: str) -> tuple[int, int]:
    """(bytes, n_safetensors) for `model_name` under `hf_home`. (0, 0) if absent."""
    d = hf_home / "hub" / ("models--" + model_name.replace("/", "--"))
    if not d.is_dir():
        return 0, 0
    files = list(d.rglob("*"))
    size = sum(f.stat().st_size for f in files if f.is_file())
    n_st = len([f for f in files if f.name.endswith(".safetensors")])
    return size, n_st


def resolve_hf_home(model_name: str) -> Path | None:
    """Pick the HF cache that actually HAS the model, and warn if that isn't $HF_HOME.

    Why this exists: on Vast, `HF_HOME` defaults to `${WORKSPACE}/.hf_home`, but a Jupyter kernel
    can be started with a different environment than your login shell. On 2026-07-21 that split
    caused a 15 GB re-download of an already-cached model straight into a full disk. A cache miss
    on a gated 15 GB model should never be silent.

    Returns the chosen cache root, or None to let huggingface_hub decide.
    """
    candidates = []
    if hh := os.getenv("HF_HOME"):
        candidates.append(Path(hh))
    candidates += [Path.home() / ".cache" / "huggingface", Path("/root/.cache/huggingface")]

    seen, scored = set(), []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        size, n_st = _model_cache_size(c, model_name)
        scored.append((c, size, n_st))

    best = max(scored, key=lambda x: x[1])
    if best[1] == 0:
        print(f"⚠ {model_name} is not cached anywhere obvious — expect a multi-GB download.")
        for c, size, _ in scored:
            print(f"    checked {c} ({size / 1e9:.1f} GB)")
        return None

    current = Path(os.getenv("HF_HOME")) if os.getenv("HF_HOME") else None
    if current is not None and current != best[0]:
        cur_size = next((s for c, s, _ in scored if c == current), 0)
        print(f"⚠ HF_HOME={current} has only {cur_size / 1e9:.1f} GB of {model_name}; "
              f"using {best[0]} ({best[1] / 1e9:.1f} GB, {best[2]} safetensors) instead "
              f"to avoid re-downloading.")
    return best[0]


def init(
    model_name: str = MODEL_NAME,
    seed: int = SEED,
    load_in_8bit: bool = True,
    project_root: Path | None = None,
    hf_token: str | None = None,
) -> None:
    """Load the model and tokenizer ONCE and seed the RNGs.

    8-bit is what [NB] sections 4-5 use and what fits alongside cached activations on the GPU
    box. Probes and steering operate on hidden states, which stay fp16 under 8-bit
    quantization, so quantization does not change the probing interface.

    Idempotent: a second call with the same model_name is a no-op.
    """
    global model, tokenizer, N_LAYERS, D_MODEL, MODEL_NAME, SEED, PROJECT_ROOT

    if project_root is not None:
        PROJECT_ROOT = Path(project_root).resolve()

    SEED = seed
    t.manual_seed(seed)
    np.random.seed(seed)

    if model is not None and MODEL_NAME == model_name:
        return  # already loaded

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    MODEL_NAME = model_name

    # Llama-3.1-8B-Instruct is a GATED repo: without a token `from_pretrained` fails with a
    # GatedRepoError 401, which reads like a model-name problem but is purely auth.
    env_path = load_env()
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not found — meta-llama/Llama-3.1-8B-Instruct is a gated repo and will 401 "
            "without it.\n"
            f"Looked for: {PROJECT_ROOT / '.env'}"
            + (f"\nLoaded {env_path}, but it has no HF_TOKEN." if env_path
               else " — but no .env is there.")
            + f"\nFix: create {PROJECT_ROOT / '.env'} with HF_TOKEN=... (and ANTHROPIC_API_KEY=... "
              "for the judge), or export HF_TOKEN in the shell."
        )
    if env_path:
        print(f"env: {env_path}")

    # Point HF at the cache that actually holds the weights (see resolve_hf_home).
    cache_root = resolve_hf_home(model_name)
    hub_kwargs = {"cache_dir": str(cache_root / "hub")} if cache_root else {}

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, **hub_kwargs)
    kwargs = {"device_map": "auto", "token": token, **hub_kwargs}
    if load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Right padding + eos-as-pad: both the pooled and full-sequence extractors assume the real
    # tokens are a left-aligned prefix (they index the last real token as
    # attention_mask.sum() - 1). Changing padding_side would silently break that indexing.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    N_LAYERS = model.config.num_hidden_layers
    D_MODEL = model.config.hidden_size
    print(f"{model_name}: {N_LAYERS} layers, d_model={D_MODEL}, device={model.device}")


def get_model():
    if model is None:
        _require("model")
    return model


def get_tokenizer():
    if tokenizer is None:
        _require("tokenizer")
    return tokenizer
