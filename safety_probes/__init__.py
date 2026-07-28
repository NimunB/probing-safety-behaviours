"""safety_probes — linear probes for safety-relevant behaviours (deception, high-stakes).

One reusable pipeline, run identically on multiple behaviours, that (1) selects a detection
layer and a steering layer-block by different, purpose-matched criteria, (2) shows linear
structure, (3) trains a grid of probes and evaluates them in-distribution and OOD, and (4)
tests the probe directions causally by steering, with proper controls.

Extracted from `safety_probes.ipynb`. See README.md for extraction status — this package is
NOT yet complete; datasets, steering, judge, and viz still live in the notebook.

Quick start:

    from safety_probes import runtime, chat, extraction, probes
    runtime.init()
    text, mask = chat.chat_with_assistant_mask(messages)
"""

from . import (
    chat,
    datasets,
    evaluation,
    export,
    extraction,
    judge,
    metrics,
    probes,
    runtime,
    spec,
    steering,
    summary,
    viz,
)
from .spec import BehaviourSpec

__all__ = [
    "runtime",
    "chat",
    "probes",
    "extraction",
    "datasets",
    "metrics",
    "steering",
    "judge",
    "evaluation",
    "summary",
    "viz",
    "export",
    "spec",
    "BehaviourSpec",
]

__version__ = "0.1.0.dev0"
