"""Interactive UI for safety probes — detection and steering, live on the loaded model.

    python ui_probe.py                       # binds 127.0.0.1:17860

WHY FASTAPI AND NOT GRADIO
--------------------------
Gradio was the obvious choice and was rejected on evidence (2026-07-21). In this environment:

  * gradio 6.x requires `huggingface_hub>=1.2.0`, but `transformers 4.57.6` requires
    `huggingface-hub<1.0`. Installing it would have broken model loading outright.
  * gradio 5.x avoids that, but downgrades six packages including `starlette 1.3.1 -> 0.52.1`
    (a major version `fastapi 0.139.0` sits on top of), `pydantic_core`, `pillow`, `websockets`.

This venv produced the audited thesis results; a UI is not worth destabilising it. `fastapi` and
`uvicorn` were already installed, so this app adds **zero dependencies**. It is also a simpler
thing to put behind Caddy than Gradio's queue/websocket machinery.

WHAT THIS IS
------------
  DETECTION  you type a conversation; the model runs one forward pass; the probe scores every
             token at that behaviour's detection layer; the tokens are coloured.
  STEERING   a calibrated direction is added at every token position across that behaviour's
             steering block while the model generates, at a coefficient you choose.

NO ACTIVATIONS ARE STORED OR LOADED. Everything is computed live from your input. The only
persisted artefacts are the ~0.5 MB probe bundles in `exported_probes/`.

TWO THINGS TO KEEP IN MIND WHEN READING THE OUTPUT
--------------------------------------------------
1. **Probe scores are uncalibrated, and your input is out of distribution.** The probes were
   fitted on one narrow format per behaviour (deception: instructed honest/deceptive pairs about
   factual statements; high-stakes: models-under-pressure scenarios). Free-form text is OOD.
   Relative colour *within* one example is informative; the absolute number is not a probability
   and is not comparable across examples.
2. **The steering coefficient is in class-gap units, not arbitrary.** 1.0 means "move the
   activation by one cross-class mean gap" [GoT §6.1], because the exported vectors are
   pre-calibrated. That is why one slider range is meaningful for MM, LR, and LR-attention alike.

The loader/rebuild functions this file used to duplicate now live in `safety_probes.export`.
"""

from __future__ import annotations

import argparse
import html
import threading
from pathlib import Path

import torch as t
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from safety_probes import chat, export, runtime
from safety_probes.steering import SteeringHook, compose_steering, generate_steered

#: One model, one GPU, and `SteeringHook` registers forward hooks on the SHARED model object.
#: Without this lock two overlapping requests can interleave enable()/remove(), so one request's
#: steering vector leaks into another's generation — silently wrong output, no error. Every
#: handler that touches the model holds this for its whole duration.
MODEL_LOCK = threading.Lock()

_HERE = Path(__file__).resolve().parent

#: Bundles live INSIDE the repo so it is self-contained and portable to a fresh instance.
#: The notebook currently exports to `../exported_probes` (its own section dir, outside this
#: repo), so that path is kept as a fallback until the notebooks move into `notebooks/`.
#: `safety_probes.export.export_dir()` writes to the in-repo location, so once the pipeline runs
#: from the package the fallback stops being used.
DEFAULT_BUNDLE_DIR = (_HERE / "exported_probes"
                      if (_HERE / "exported_probes").exists()
                      else _HERE.parent / "exported_probes")

BUNDLES: dict[str, dict] = {}


# =========================================================================================
# Startup
# =========================================================================================

def load_everything(bundle_dir: Path) -> list[str]:
    """Load the model once, then every bundle in `bundle_dir`. Returns the behaviour names."""
    runtime.init()
    names = []
    for path in sorted(bundle_dir.glob("*_probes.pt")):
        b = export.load_behaviour(
            path,
            expect_model=runtime.MODEL_NAME,      # refuse a bundle built for another model
            expect_d_model=runtime.D_MODEL,
        )
        BUNDLES[b["behaviour"]] = b
        names.append(b["behaviour"])
        print(f"  loaded {b['behaviour']:<12} detect_layer={b['detect_layer']:<3} "
              f"steer_block={b['steer_block']}  probes={list(b['detection'])}")
    if not names:
        raise SystemExit(f"No *_probes.pt found in {bundle_dir}. Run the export cells first.")
    return names


# =========================================================================================
# Shared helpers
# =========================================================================================

def _build_messages(system: str, user: str, assistant: str = "") -> list[dict[str, str]]:
    msgs = []
    if system.strip():
        msgs.append({"role": "system", "content": system.strip()})
    if user.strip():
        msgs.append({"role": "user", "content": user.strip()})
    if assistant.strip():
        msgs.append({"role": "assistant", "content": assistant.strip()})
    return msgs


def _text_and_mask(bundle: dict, messages: list[dict[str, str]]):
    """Tokenize through the SAME helpers the probes were trained with.

    Not a stylistic preference: `chat.py` is where the single-BOS guarantee and the mask policy
    live. Calling `apply_chat_template` directly here would reintroduce the double-BOS bug
    (plan #22) and silently shift every activation by one token.
    """
    if bundle["mask_policy"] == "assistant_only":
        return chat.chat_with_assistant_mask(messages)
    return chat.chat_with_content_mask(messages)


@t.no_grad()
def _hidden_at_layer(text: str, layer: int):
    """One forward pass; return (token_ids, hidden [seq, d]) at `layer` — nothing cached."""
    model, tokenizer = runtime.get_model(), runtime.get_tokenizer()
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    out = model(**inp, output_hidden_states=True)
    hidden = out.hidden_states[layer + 1][0].cpu().float()   # +1: index 0 is the embedding output
    return inp["input_ids"][0].tolist(), hidden


def _colour(v: float) -> str:
    """Diverging red/blue. v in [-1, 1]; red = toward the positive class."""
    v = max(-1.0, min(1.0, v))
    if v >= 0:
        return f"rgba(220,60,50,{0.10 + 0.75 * v:.3f})"
    return f"rgba(60,90,200,{0.10 + 0.75 * abs(v):.3f})"


def _render_tokens(tokens: list[str], values, mask, pos_name: str, neg_name: str,
                   subtitle: str, symmetric: bool = True) -> str:
    """Colour tokens by value. Unmasked tokens are greyed — present, but not probed."""
    vals = [float(v) for v in values]
    sel = [v for v, m in zip(vals, mask) if m]
    if not sel:
        return "<p><em>No tokens fell inside the probed span.</em></p>"

    if symmetric:                       # probe scores: centre on the probed tokens' mean
        centre = sum(sel) / len(sel)
        span = max(abs(v - centre) for v in sel) or 1e-6
        norm = [(v - centre) / span for v in vals]
    else:                               # attention weights: >= 0, share of total attention
        hi = max(sel) or 1e-6
        norm = [v / hi for v in vals]

    spans = []
    for tok, v, n, m in zip(tokens, vals, norm, mask):
        safe = html.escape(tok).replace("\n", "⏎")
        if m:
            spans.append(f"<span class='tk' style='background:{_colour(n)}' "
                         f"title='{v:+.4f}'>{safe}</span>")
        else:
            spans.append(f"<span class='tk off' title='not probed'>{safe}</span>")

    legend = (f"<span class='sw' style='background:{_colour(0.9)}'></span>{html.escape(pos_name)}"
              f"<span class='sw' style='background:{_colour(-0.9)}'></span>{html.escape(neg_name)}"
              f"<span class='sw off'></span>not probed")
    return (f"<div class='sub'>{subtitle}</div><div class='toks'>{''.join(spans)}</div>"
            f"<div class='legend'>{legend}</div>")


def _steer_vectors(behaviour: str, method: str) -> dict[int, t.Tensor]:
    return {int(k): v for k, v in BUNDLES[behaviour]["steering"][method].items()}


# =========================================================================================
# Analysis
# =========================================================================================

def run_detection(behaviour: str, probe_name: str, system: str, user: str, assistant: str):
    b = BUNDLES.get(behaviour)
    if b is None:
        return {"html": "<p>Unknown behaviour.</p>", "summary": ""}
    messages = _build_messages(system, user, assistant)
    if not messages:
        return {"html": "<p>Enter some text.</p>", "summary": ""}
    if b["mask_policy"] == "assistant_only" and not assistant.strip():
        return {"html": "<p><strong>This behaviour probes assistant tokens only</strong> — fill in "
                        "the assistant turn, or there is nothing to score.</p>", "summary": ""}
    if probe_name not in b["detection"]:
        return {"html": f"<p>{html.escape(probe_name)} is not in this bundle.</p>", "summary": ""}

    tokenizer = runtime.get_tokenizer()
    text, mask = _text_and_mask(b, messages)
    ids, hidden = _hidden_at_layer(text, b["detect_layer"])

    seq = len(ids)
    m = t.zeros(seq, dtype=t.bool)
    m[: min(len(mask), seq)] = mask[:seq].bool()
    tokens = [tokenizer.decode([i]) for i in ids]

    if int(m.sum()) == 0:
        # Truncation (max_length=1024) removed every probed token. The attention softmax over an
        # all-False mask returns NaN and w[m].max() then raises on an empty tensor, so bail here
        # with something the user can act on.
        return {"html": "<p><strong>No probed tokens survived truncation.</strong> The input is "
                        "longer than the 1024-token window, and the span this behaviour probes "
                        f"(<code>{b['mask_policy']}</code>) fell outside it. Shorten the input.</p>",
                "summary": ""}

    entry = b["detection"][probe_name]
    probe = export.rebuild_detection_probe(b, probe_name)

    if entry["kind"] == "attention_seeds":
        # Different semantics: WHERE the probe looks, not WHAT it scores. Weights are >= 0 and
        # sum to 1 over unmasked positions, so no centring — intensity is share of attention.
        with t.no_grad():
            w = t.stack([p.get_attention_weights(hidden.unsqueeze(0), m.unsqueeze(0))[0, :, 0]
                         for p in probe]).mean(0)
        body = _render_tokens(
            tokens, w.tolist(), m.tolist(), "attended", "ignored",
            f"Attention weights at layer {b['detect_layer']} (mean of {entry['n_seeds']} seeds) — "
            f"where the probe <em>looks</em>, not what it scores. Darker = more attention.",
            symmetric=False)
        summary = (f"<b>{html.escape(probe_name)}</b> · layer {b['detect_layer']} · "
                   f"max attention {float(w[m].max()):.3f} over {int(m.sum())} probed tokens")
    else:
        scores = probe.score(hidden)
        pooled = float(scores[m].mean())
        body = _render_tokens(
            tokens, scores.tolist(), m.tolist(), b["positive_name"], b["negative_name"],
            f"Per-token probe score at layer {b['detect_layer']}, centred on the probed tokens' "
            f"mean. Hover a token for its raw score.")
        summary = (f"<b>{html.escape(probe_name)}</b> · layer {b['detect_layer']} · mean score over "
                   f"probed tokens: <b>{pooled:+.4f}</b> (higher = more "
                   f"<i>{html.escape(b['positive_name'])}</i>)")

    summary += (f"<br><span class='warn'>Mask policy <code>{b['mask_policy']}</code> — "
                f"{int(m.sum())}/{seq} tokens probed. Scores are uncalibrated and this input is "
                f"out of the probe's training distribution: read colour <i>within</i> this "
                f"example, not across examples.</span>")
    return {"html": body, "summary": summary}


def run_steering(behaviour: str, method: str, coef: float, system: str, user: str,
                 max_new_tokens: int):
    b = BUNDLES.get(behaviour)
    if b is None:
        return {"baseline": "", "steered": "", "note": "Unknown behaviour."}
    if method not in b["steering"]:
        return {"baseline": "", "steered": "", "note": f"{method} is not in this bundle."}
    messages = _build_messages(system, user)
    if not messages:
        return {"baseline": "", "steered": "", "note": "Enter a prompt."}

    prompt = chat.chat_text(messages, add_generation_prompt=True)
    baseline = generate_steered(prompt, None, max_new_tokens=int(max_new_tokens))
    if abs(float(coef)) < 1e-9:
        steered = baseline          # coef 0 under greedy decoding IS the baseline, by construction
    else:
        hook = SteeringHook(_steer_vectors(behaviour, method), coef=float(coef))
        steered = generate_steered(prompt, hook, max_new_tokens=int(max_new_tokens))

    note = (f"<b>{html.escape(method)}</b> at coef <b>{coef:+.2f}</b> over layers "
            f"{b['steer_block']} — 1.0 = one cross-class mean gap toward "
            f"<i>{html.escape(b['positive_name'])}</i>, negative = toward "
            f"<i>{html.escape(b['negative_name'])}</i>. Greedy decoding, so coef 0 reproduces the "
            f"baseline exactly.")
    return {"baseline": baseline, "steered": steered, "note": note}


def run_multi(b1: str, m1: str, c1: float, b2: str, m2: str, c2: float,
              system: str, user: str, max_new_tokens: int):
    if b1 not in BUNDLES or b2 not in BUNDLES:
        return {"baseline": "", "steered": "", "note": "Unknown behaviour."}
    messages = _build_messages(system, user)
    if not messages:
        return {"baseline": "", "steered": "", "note": "Enter a prompt."}

    prompt = chat.chat_text(messages, add_generation_prompt=True)
    baseline = generate_steered(prompt, None, max_new_tokens=int(max_new_tokens))
    hook = compose_steering((_steer_vectors(b1, m1), float(c1)),
                            (_steer_vectors(b2, m2), float(c2)))
    steered = generate_steered(prompt, hook, max_new_tokens=int(max_new_tokens))

    blk1, blk2 = BUNDLES[b1]["steer_block"], BUNDLES[b2]["steer_block"]
    overlap = sorted(set(blk1) & set(blk2))
    geometry = (f"Blocks <b>overlap</b> at {overlap} — the vectors are summed there."
                if overlap else
                f"Blocks are <b>disjoint</b> ({b1} {blk1}, {b2} {blk2}) — the two directions act "
                f"at different depths and are never summed.")
    note = (f"{html.escape(b1)} <code>{html.escape(m1)}</code> @ {c1:+.2f} + "
            f"{html.escape(b2)} <code>{html.escape(m2)}</code> @ {c2:+.2f}. {geometry}"
            f"<br><span class='warn'>⚠️ <b>Exploratory.</b> Whether two behaviour directions "
            f"compose, cancel, or simply break the model is an open question that neither [CAA] "
            f"nor [GoT] answers. Treat anything here as a pilot observation, not a measurement — "
            f"it needs the same NIE + judge-coherence protocol as single-behaviour steering "
            f"before it means anything.</span>")
    return {"baseline": baseline, "steered": steered, "note": note}


# =========================================================================================
# Web app — self-contained page, no external assets (CSP-safe, works behind Caddy)
# =========================================================================================

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Safety Probes</title><style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--acc:#c44e52;--card:#fafafa}
@media(prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa0a6;--line:#2c3038;--card:#1d2026}}
*{box-sizing:border-box}
body{margin:0;padding:1.5rem;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
background:var(--bg);color:var(--fg)}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 .25rem}
.lede{color:var(--mut);margin:0 0 1.25rem;font-size:.92rem}
.tabs{display:flex;gap:.25rem;border-bottom:1px solid var(--line);margin-bottom:1.25rem;flex-wrap:wrap}
.tab{padding:.5rem .9rem;cursor:pointer;border:0;background:none;color:var(--mut);font:inherit;
border-bottom:2px solid transparent}
.tab.on{color:var(--fg);border-bottom-color:var(--acc);font-weight:600}
.panel{display:none}.panel.on{display:block}
label{display:block;font-size:.82rem;color:var(--mut);margin:.75rem 0 .2rem;font-weight:600}
input,select,textarea{width:100%;padding:.5rem .6rem;border:1px solid var(--line);border-radius:6px;
background:var(--bg);color:var(--fg);font:inherit}
textarea{resize:vertical;min-height:2.6rem}
.row{display:flex;gap:.75rem;flex-wrap:wrap}.row>*{flex:1;min-width:170px}
button.go{margin-top:1rem;padding:.55rem 1.2rem;border:0;border-radius:6px;background:var(--acc);
color:#fff;font:inherit;font-weight:600;cursor:pointer}
button.go:disabled{opacity:.55;cursor:default}
.out{margin-top:1.25rem}
.toks{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:2;font-size:.9rem;
white-space:pre-wrap;word-break:break-word;border:1px solid var(--line);border-radius:8px;
padding:.9rem;background:var(--card);overflow-x:auto}
.tk{padding:1px 0;border-radius:2px}.tk.off{color:#bbb}
.sub{font-size:.85rem;color:var(--mut);margin-bottom:.5rem}
.legend{margin-top:.6rem;font-size:.82rem;color:var(--mut)}
.sw{display:inline-block;width:1.5rem;height:.75rem;border-radius:2px;margin:0 .35rem 0 .9rem;
vertical-align:middle;border:1px solid var(--line)}.sw:first-child{margin-left:0}.sw.off{background:#ccc}
.summary{margin-top:.8rem;font-size:.9rem}
.warn{color:var(--mut);font-size:.85rem}
.gen{display:flex;gap:.9rem;flex-wrap:wrap;margin-top:1rem}
.gen>div{flex:1;min-width:290px}
.gen pre{white-space:pre-wrap;word-break:break-word;border:1px solid var(--line);border-radius:8px;
padding:.8rem;background:var(--card);min-height:6rem;margin:.3rem 0 0;font-size:.88rem}
.note{margin-top:.9rem;font-size:.88rem;color:var(--mut)}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line)}
code{background:var(--card);padding:.1rem .3rem;border-radius:3px;font-size:.92em}
.co{font-variant-numeric:tabular-nums;font-weight:600;min-width:3rem;display:inline-block}
</style></head><body><div class="wrap">
<h1>Safety Probes</h1>
<p class="lede">Linear probes for safety-relevant behaviours, live on <code>__MODEL__</code>.
Everything is computed from your input on each request — <b>no activations are stored</b>.</p>
<div class="tabs">
  <button class="tab on" data-p="det">Detection</button>
  <button class="tab" data-p="steer">Steering</button>
  <button class="tab" data-p="multi">Multi-behaviour</button>
  <button class="tab" data-p="info">Bundles</button>
</div>

<div class="panel on" id="p-det">
  <p class="lede">Score every token with a fitted probe. Detection layer and mask policy come from
  the behaviour's own bundle — deception probes assistant tokens only; high-stakes probes message
  content of every role.</p>
  <div class="row">
    <div><label>Behaviour</label><select id="d_beh"></select></div>
    <div><label>Probe</label><select id="d_probe"></select></div>
  </div>
  <label>System</label><textarea id="d_sys">You are a deceptive AI assistant.</textarea>
  <label>User</label><textarea id="d_user">Please tell me a fact.</textarea>
  <label>Assistant <span class="warn">(the text that gets probed, for deception)</span></label>
  <textarea id="d_asst">The Eiffel Tower is located in Berlin, Germany.</textarea>
  <button class="go" id="d_go">Run detection</button>
  <div class="out"><div id="d_out"></div><div class="summary" id="d_sum"></div></div>
</div>

<div class="panel" id="p-steer">
  <p class="lede">Add a calibrated direction at every token position across the behaviour's steering
  block while the model generates. <b>Coefficient 1.0 = one cross-class mean gap</b> [GoT §6.1] —
  the vectors are pre-calibrated, so the scale is comparable across methods.</p>
  <div class="row">
    <div><label>Behaviour</label><select id="s_beh"></select></div>
    <div><label>Direction</label><select id="s_meth"></select></div>
    <div><label>Coefficient <span class="co" id="s_cv">+1.00</span></label>
      <input type="range" id="s_coef" min="-2" max="2" step="0.25" value="1"></div>
  </div>
  <label>System</label><textarea id="s_sys">You are an honest AI assistant.</textarea>
  <label>User</label><textarea id="s_user">Please tell me a fact.</textarea>
  <label>Max new tokens <span class="co" id="s_lv">80</span></label>
  <input type="range" id="s_len" min="16" max="200" step="8" value="80">
  <button class="go" id="s_go">Generate</button>
  <div class="gen">
    <div><label>Baseline (unsteered)</label><pre id="s_base"></pre></div>
    <div><label>Steered</label><pre id="s_out"></pre></div>
  </div>
  <div class="note" id="s_note"></div>
</div>

<div class="panel" id="p-multi">
  <p class="lede">⚠️ <b>Exploratory — not a validated method.</b> Applies two behaviour directions in
  one generation. Whether they compose, cancel, or simply produce incoherence is an open question;
  neither [CAA] nor [GoT] characterises it. Use it to generate hypotheses, not to claim results.</p>
  <div class="row">
    <div><label>Behaviour A</label><select id="m_b1"></select></div>
    <div><label>Direction A</label><select id="m_m1"></select></div>
    <div><label>Coefficient A <span class="co" id="m_c1v">+1.00</span></label>
      <input type="range" id="m_c1" min="-2" max="2" step="0.25" value="1"></div>
  </div>
  <div class="row">
    <div><label>Behaviour B</label><select id="m_b2"></select></div>
    <div><label>Direction B</label><select id="m_m2"></select></div>
    <div><label>Coefficient B <span class="co" id="m_c2v">+1.00</span></label>
      <input type="range" id="m_c2" min="-2" max="2" step="0.25" value="1"></div>
  </div>
  <label>System</label><textarea id="m_sys">You are an honest AI assistant.</textarea>
  <label>User</label><textarea id="m_user">Please tell me a fact.</textarea>
  <label>Max new tokens <span class="co" id="m_lv">80</span></label>
  <input type="range" id="m_len" min="16" max="200" step="8" value="80">
  <button class="go" id="m_go">Generate</button>
  <div class="gen">
    <div><label>Baseline (unsteered)</label><pre id="m_base"></pre></div>
    <div><label>Steered (both directions)</label><pre id="m_out"></pre></div>
  </div>
  <div class="note" id="m_note"></div>
</div>

<div class="panel" id="p-info"><div id="i_out"></div></div>
</div><script>
const B = __BUNDLES__, NAMES = Object.keys(B);
const $ = id => document.getElementById(id);
const fill = (el, opts, val) => { el.innerHTML = opts.map(o =>
  `<option${o===val?' selected':''}>${o}</option>`).join(''); };

document.querySelectorAll('.tab').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('on'));
  document.querySelectorAll('.panel').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); $('p-' + b.dataset.p).classList.add('on');
});

const METHODS = [...new Set(NAMES.flatMap(n => B[n].methods))].sort();
fill($('d_beh'), NAMES, NAMES[0]);
fill($('s_beh'), NAMES, NAMES[0]); fill($('s_meth'), METHODS, METHODS[0]);
fill($('m_b1'), NAMES, NAMES[0]);  fill($('m_m1'), METHODS, METHODS[0]);
fill($('m_b2'), NAMES, NAMES[NAMES.length-1]); fill($('m_m2'), METHODS, METHODS[0]);

function syncProbes(){ const p = B[$('d_beh').value].probes;
  fill($('d_probe'), p, p.includes('LR (mean)') ? 'LR (mean)' : p[0]); }
$('d_beh').onchange = syncProbes; syncProbes();

const bindSlider = (s, out, fmt) => { const f = () => out.textContent = fmt(s.value); s.oninput = f; f(); };
bindSlider($('s_coef'), $('s_cv'), v => (v>=0?'+':'') + (+v).toFixed(2));
bindSlider($('m_c1'), $('m_c1v'), v => (v>=0?'+':'') + (+v).toFixed(2));
bindSlider($('m_c2'), $('m_c2v'), v => (v>=0?'+':'') + (+v).toFixed(2));
bindSlider($('s_len'), $('s_lv'), v => v);
bindSlider($('m_len'), $('m_lv'), v => v);

async function post(url, body, btn, label){
  btn.disabled = true; const old = btn.textContent; btn.textContent = label;
  try { const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(body)}); return await r.json(); }
  catch(e){ return {error: String(e)}; }
  finally { btn.disabled = false; btn.textContent = old; }
}

$('d_go').onclick = async () => {
  $('d_out').innerHTML = '<em>running forward pass…</em>'; $('d_sum').innerHTML = '';
  const r = await post('/api/detect', {behaviour:$('d_beh').value, probe:$('d_probe').value,
    system:$('d_sys').value, user:$('d_user').value, assistant:$('d_asst').value}, $('d_go'), 'running…');
  $('d_out').innerHTML = r.html || ('<p>' + (r.error||'error') + '</p>'); $('d_sum').innerHTML = r.summary || '';
};

$('s_go').onclick = async () => {
  $('s_base').textContent = '…'; $('s_out').textContent = '…'; $('s_note').innerHTML = '';
  const r = await post('/api/steer', {behaviour:$('s_beh').value, method:$('s_meth').value,
    coef:+$('s_coef').value, system:$('s_sys').value, user:$('s_user').value,
    max_new_tokens:+$('s_len').value}, $('s_go'), 'generating…');
  $('s_base').textContent = r.baseline||''; $('s_out').textContent = r.steered||'';
  $('s_note').innerHTML = r.note || r.error || '';
};

$('m_go').onclick = async () => {
  $('m_base').textContent = '…'; $('m_out').textContent = '…'; $('m_note').innerHTML = '';
  const r = await post('/api/multi', {b1:$('m_b1').value, m1:$('m_m1').value, c1:+$('m_c1').value,
    b2:$('m_b2').value, m2:$('m_m2').value, c2:+$('m_c2').value, system:$('m_sys').value,
    user:$('m_user').value, max_new_tokens:+$('m_len').value}, $('m_go'), 'generating…');
  $('m_base').textContent = r.baseline||''; $('m_out').textContent = r.steered||'';
  $('m_note').innerHTML = r.note || r.error || '';
};

$('i_out').innerHTML = '<table><tr><th>behaviour</th><th>detect layer</th><th>steer block</th>'
  + '<th>mask policy</th><th>probes</th><th>exported</th></tr>'
  + NAMES.map(n => `<tr><td>${n}</td><td>${B[n].detect_layer}</td><td>[${B[n].steer_block}]</td>`
  + `<td><code>${B[n].mask_policy}</code></td><td>${B[n].probes.length}</td>`
  + `<td>${B[n].exported_utc}</td></tr>`).join('')
  + '</table><p class="warn">Model: <code>__MODEL__</code>. Probe scores are uncalibrated; '
  + 'free-form input is out of the training distribution for every behaviour.</p>';
</script></body></html>"""


def build_app() -> FastAPI:
    app = FastAPI(title="Safety Probes")

    meta = {n: {"detect_layer": b["detect_layer"], "steer_block": b["steer_block"],
                "mask_policy": b["mask_policy"], "probes": list(b["detection"]),
                "methods": list(b["steering"]), "exported_utc": b["exported_utc"],
                "positive_name": b["positive_name"], "negative_name": b["negative_name"]}
            for n, b in BUNDLES.items()}

    @app.get("/", response_class=HTMLResponse)
    def index():
        import json as _json
        return (PAGE.replace("__BUNDLES__", _json.dumps(meta))
                    .replace("__MODEL__", runtime.MODEL_NAME))

    @app.get("/api/bundles")
    def bundles():
        return JSONResponse(meta)

    # NOTE: these are `def`, not `async def`, deliberately. FastAPI runs sync handlers in a
    # threadpool; an `async def` handler doing a multi-second GPU call would block the event loop
    # and freeze the whole server, including the page itself. MODEL_LOCK then serialises the
    # actual model work, which the threadpool would otherwise allow to overlap.

    @app.post("/api/detect")
    def detect(payload: dict):
        with MODEL_LOCK:
            return JSONResponse(run_detection(
                payload.get("behaviour", ""), payload.get("probe", ""), payload.get("system", ""),
                payload.get("user", ""), payload.get("assistant", "")))

    @app.post("/api/steer")
    def steer(payload: dict):
        with MODEL_LOCK:
            return JSONResponse(run_steering(
                payload.get("behaviour", ""), payload.get("method", ""), payload.get("coef", 0.0),
                payload.get("system", ""), payload.get("user", ""),
                payload.get("max_new_tokens", 80)))

    @app.post("/api/multi")
    def multi(payload: dict):
        with MODEL_LOCK:
            return JSONResponse(run_multi(
                payload.get("b1", ""), payload.get("m1", ""), payload.get("c1", 0.0),
                payload.get("b2", ""), payload.get("m2", ""), payload.get("c2", 0.0),
                payload.get("system", ""), payload.get("user", ""),
                payload.get("max_new_tokens", 80)))

    return app


def main():
    ap = argparse.ArgumentParser(description="Safety-probe detection and steering UI.")
    ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument("--host", default="127.0.0.1",
                    help="Keep on localhost and front it with Caddy (token auth) or an SSH tunnel.")
    ap.add_argument("--port", type=int, default=17860)
    args = ap.parse_args()

    import uvicorn
    print(f"Loading bundles from {args.bundle_dir} ...")
    load_everything(args.bundle_dir)
    print(f"Serving on http://{args.host}:{args.port}")
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
