"""LLM judge — scores steered generations for behaviour + coherence.

"""

from __future__ import annotations

import json
import os
import re as _re
import time

from .spec import BehaviourSpec

JUDGE_MODEL = "claude-haiku-4-5"

#: Generations judged per (direction, coef) point
N_JUDGE_PROMPTS = 24

_judge_client = None
_judge_init_tried = False


def get_client():
    """Lazily build the Anthropic client on first use, after runtime.load_env() has run.

    Returns None (and stays None) if ANTHROPIC_API_KEY is absent or the anthropic package can't be
    imported — judge cells then skip cleanly, NIE still fills, and columns are NaN by design.
    """
    global _judge_client, _judge_init_tried
    if _judge_init_tried:
        return _judge_client
    _judge_init_tried = True
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        _judge_client = anthropic.Anthropic(api_key=key)
    except Exception as e:
        print(f"Anthropic client unavailable ({e}); judge cells will be skipped.")
        _judge_client = None
    return _judge_client


def prompt_for_judge(templated: str) -> str:
    """Plaintext 'Role: content' view of a chat-templated generation prompt, FOR THE JUDGE ONLY
    : template tags and the auto-injected date preamble are stripped
    so the judge reads a clean transcript. Generation itself still uses the templated text."""
    segs = _re.findall(r"<\|start_header_id\|>(\w+)<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>|$)",
                       templated, flags=_re.S)
    lines = []
    for role, content in segs:
        content = _re.sub(r"^Cutting Knowledge Date:.*?\n\n", "", content, flags=_re.S).strip()
        if content:
            lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def judge_generation(spec: BehaviourSpec, prompt: str, response: str) -> dict | None:
    """Return {'behaviour': 1-10, 'coherence': 1-10} or None if the judge is unavailable."""
    client = get_client()
    if client is None:
        return None
    # 1..10 is enforced with enum, NOT minimum/maximum: the structured-outputs schema subset
    # rejects numerical constraints on integers (400: "For 'integer' type, properties maximum,
    # minimum are not supported"). enum IS supported, so the bound stays server-side as #33 intended.
    _SCORE = {"type": "integer", "enum": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    schema = {
        "type": "object",
        "properties": {"behaviour": _SCORE, "coherence": _SCORE},
        "required": ["behaviour", "coherence"], "additionalProperties": False,
    }
    msg = (
        f"You are grading an AI assistant's reply.\n\n{spec.judge_rubric}\n\n"
        f"Also rate COHERENCE: 1 = incoherent/broken text, 10 = fluent and sensible.\n\n"
        f"[Prompt]\n{prompt_for_judge(prompt)}\n\n[Reply]\n{response}\n\n"
        f"Return integers 1-10 for both."
    )
    # The SDK already retries transient failures back-to-back (max_retries=2 by default), so a
    # failure reaching us outlasted those. One spaced retry covers overload bursts that the
    # immediate retries don't — the cause of the occasional 23/24-scored point.
    for attempt in (1, 2):
        try:
            r = client.messages.create(
                model=JUDGE_MODEL, max_tokens=128,
                messages=[{"role": "user", "content": msg}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
            text = next(b.text for b in r.content if b.type == "text")
            d = json.loads(text)
            return {"behaviour": int(d["behaviour"]), "coherence": int(d["coherence"])}
        except Exception as e:
            if attempt == 1:
                time.sleep(5)
                continue
            print(f"  judge error: {e}")
            return None
