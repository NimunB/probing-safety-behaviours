"""`BehaviourSpec` — everything behaviour-specific, in one dataclass (notebook §1b).

Kept in its own module (rather than inside `datasets`) so that lightweight consumers — the
export bundle, the UI — can import the type without pulling in HuggingFace dataset loading.

Adding a behaviour means writing one of these plus a loader; nothing in the pipeline is
hard-coded to deception or high-stakes. Sycophancy will replace the YES/NO `readout_targets`
with A/B.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BehaviourSpec:
    """Everything behaviour-specific for the probing/steering pipeline."""

    name: str                        # "deception" | "high_stakes"
    positive_name: str               # human label for class 1, e.g. "deceptive"
    negative_name: str               # human label for class 0
    mask_policy: str                 # "assistant_only" | "content_tokens"
    readout_question: str            # appended to a transcript for the NIE readout
    readout_few_shot: str            # few-shot prefix for the readout ([GoT §6.1]-style calibration)
    readout_targets: tuple[str, str] # (positive_token, negative_token), e.g. (" YES", " NO")
    judge_rubric: str                # instruction for the LLM judge's behaviour score
    steer_coefs: tuple = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
