"""Chat templating and detection masks — the tokenization boundary of the whole pipeline.

THIS MODULE IS THE ARENA CUT. `build_detection_mask` and `clean_bpe_token` are vendored from
`ARENA_3.0/chapter1_transformer_interp/exercises/part31_linear_probes/utils.py`; after this
file exists, nothing in the project imports from the ARENA course repo.

WHY IT MATTERS MORE THAN IT LOOKS
---------------------------------
Every activation the probes ever see is indexed by a mask produced here. If templating and
masking disagree by even one token, every probe score is computed on the wrong activations —
silently, with no error. That is exactly what the double-BOS bug (plan #22) was. Any consumer
of this project — including the UI — MUST tokenize through these helpers rather than calling
`apply_chat_template` itself, or that bug returns.

VENDORING DECISION (2026-07-20)
-------------------------------
`build_detection_mask` is copied VERBATIM from ARENA's utils.py, including its double-BOS
quirk (it tokenizes its own BOS-bearing template string with add_special_tokens=True). We now
own this code and could fix it at source — but the notebook's audited behaviour is
"vendor-quirk + strip-fix in the caller", and changing the mask construction would invalidate
the line-by-line audit and every cached activation. The strip-fix therefore stays in
`chat_with_assistant_mask` where the notebook has it. Revisit only with a cache wipe and a
re-run of both behaviours.
"""

from __future__ import annotations

import torch as t
from jaxtyping import Bool
from torch import Tensor

from . import runtime


# =========================================================================================
# Vendored from ARENA utils.py — do not edit without reading the vendoring note above.
# =========================================================================================

def clean_bpe_token(tok: str) -> str:
    """Replace byte-level BPE artifacts with readable characters.

    Llama/GPT-style tokenizers use byte-level BPE where certain Unicode characters stand in
    for whitespace bytes: U+0120 for space, U+010A for newline, U+0109 for tab. This function
    replaces them so tokens are legible in plots and tables.
    """
    return tok.replace("Ġ", " ").replace("Ċ", "\\n").replace("ĉ", "\\t")


def build_detection_mask(
    messages: list[dict[str, str]],
    tokenizer,
    detect_role: str = "assistant",
) -> tuple[str, Tensor, Tensor, Tensor]:
    """
    Build a boolean detection mask for tokens belonging to detect_role's content, using the
    tokenizer's char_to_token method for robust character-to-token mapping.

    The approach:
    1. Format messages with apply_chat_template to get the full text
    2. Also format with detect_role content emptied out, and find where the two texts diverge
       (this is where the content starts)
    3. Use char_to_token() to map the content's character span to token indices

    Returns:
        Tuple of (text, token_ids [1, seq], attention_mask [1, seq], detection_mask [seq] bool).

    NOTE (vendored): assumes ONE message per role. Multi-turn transcripts need
    `chat_with_content_mask` below, which walks messages with a cursor instead.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoding = tokenizer(text, return_tensors="pt")
    seq_len = encoding["input_ids"].shape[1]

    detect_content = ""
    empty_messages = []
    for msg in messages:
        if msg["role"] == detect_role:
            detect_content = msg["content"]
            empty_messages.append({"role": msg["role"], "content": ""})
        else:
            empty_messages.append(msg)

    assert detect_content, f"No message with role={detect_role!r} found"

    empty_text = tokenizer.apply_chat_template(empty_messages, tokenize=False)

    content_start_char = 0
    for i in range(min(len(empty_text), len(text))):
        if empty_text[i] != text[i]:
            content_start_char = i
            break
    content_end_char = content_start_char + len(detect_content)

    assert text[content_start_char:content_end_char] == detect_content, (
        f"Content mismatch at char {content_start_char}: "
        f"expected {detect_content!r}, found {text[content_start_char:content_end_char]!r}"
    )

    start_tok = encoding.char_to_token(0, content_start_char)
    assert start_tok is not None, f"char_to_token returned None for start char {content_start_char}"

    last_content_tok = encoding.char_to_token(0, content_end_char - 1)
    end_tok = (last_content_tok + 1) if last_content_tok is not None else seq_len

    detection_mask = t.zeros(seq_len, dtype=t.bool)
    detection_mask[start_tok:end_tok] = True

    return text, encoding["input_ids"], encoding["attention_mask"], detection_mask


# =========================================================================================
# Project helpers (from notebook §1c) — every templated string must go through chat_text.
# =========================================================================================

def chat_text(messages, **kwargs) -> str:
    """apply_chat_template(tokenize=False) with the template's leading BOS stripped, so downstream
    tokenizer() calls (which add their own BOS) produce exactly ONE <|begin_of_text|>. Fix
    2026-07-13 (plan #22): previously every templated path double-BOSed (quirk inherited from [NB],
    whose utils tokenize the BOS-bearing template string with add_special_tokens=True). ALL
    templated-text creation must go through here so masks and activations stay aligned."""
    tok = runtime.get_tokenizer()
    return tok.apply_chat_template(messages, tokenize=False, **kwargs).removeprefix(tok.bos_token)


def chat_with_assistant_mask(messages: list[dict[str, str]]) -> tuple[str, Bool[Tensor, " seq"]]:
    """Format a chat and return (text, token-level detection mask over assistant content).
    build_detection_mask tokenizes its own template string WITH an added BOS (double-BOS);
    we strip the string's BOS and drop the mask's first position so the returned pair is aligned
    with the single-BOS tokenization every downstream tokenizer(text) call now produces.

    This is the DECEPTION mask policy ("assistant_only"): the probe must not see the system
    prompt, which literally states the honest/deceptive persona — i.e. the label.
    """
    tok = runtime.get_tokenizer()
    text, _tokens, _attn, det_mask = build_detection_mask(messages, tok)
    if text.startswith(tok.bos_token):
        text = text.removeprefix(tok.bos_token)
        det_mask = det_mask[1:]
    return text, det_mask


def chat_with_content_mask(messages: list[dict[str, str]]) -> tuple[str, Bool[Tensor, " seq"]]:
    """(text, mask over MESSAGE-CONTENT tokens of every role) — the high-stakes mask policy
    "content_tokens" (2026-07-19, plan #31; replaces full-sequence pooling, a disclosed deviation
    from [HS §2.1]). Excludes BOS, header tags, <|eot_id|>s, and the template's auto-injected date
    preamble; includes system/user/assistant contents of every turn. Multi-turn safe, unlike
    build_detection_mask (which assumes one message per role): walks the messages in template
    order with a cursor and maps each content's char span via char_to_token. NOTE: the Llama
    template TRIMS content before inserting it, hence the .strip()."""
    tok = runtime.get_tokenizer()
    text = chat_text(messages)
    enc = tok(text, return_tensors="pt")
    seq_len = enc["input_ids"].shape[1]
    mask = t.zeros(seq_len, dtype=t.bool)
    cursor = 0
    for msg in messages:
        content = msg["content"].strip()
        if not content:
            continue
        start = text.find(content, cursor)
        assert start != -1, f"message content not found in template: {content[:60]!r}"
        end = start + len(content)
        st = enc.char_to_token(0, start)
        et = enc.char_to_token(0, end - 1)
        mask[st:(et + 1) if et is not None else seq_len] = True
        cursor = end
    return text, mask
