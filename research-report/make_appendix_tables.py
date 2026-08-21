#!/usr/bin/env python
"""Generate appendix_tables.tex from exported_probes/*.

Run from research-report/ after any pipeline re-run:

    ../.venv/bin/python make_appendix_tables.py     # or /venv/main/bin/python

Writes appendix_tables.tex, which the report \\inputs. Regenerating after a re-run is the
whole point: the appendix numbers then cannot drift from the artefacts they came from.
"""
from __future__ import annotations
import csv, json, pathlib, re

EXPORT = pathlib.Path(__file__).resolve().parent.parent / "exported_probes"
OUT = pathlib.Path(__file__).resolve().parent / "appendix_tables.tex"

BEHAVIOURS = [("deception", "Deception"), ("high-stakes", "High-stakes"), ("sycophancy", "Sycophancy")]


def esc(s: str) -> str:
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
             .replace("#", r"\#").replace("<", r"$<$").replace(">", r"$>$"))


def detection_table(slug: str, title: str) -> str:
    p = EXPORT / f"{slug}_detection.csv"
    if not p.is_file():
        return f"\n\\noindent\\textcolor{{red}}{{[{esc(p.name)} not found --- re-run the pipeline]}}\n"
    rows = list(csv.DictReader(p.open()))
    evals = list(dict.fromkeys(r["eval"] for r in rows))
    probes = list(dict.fromkeys(r["probe"] for r in rows))
    by = {(r["probe"], r["eval"]): r for r in rows}

    cap = (r"\caption{%s --- detection. AUROC with bootstrap 95\%% CI; "
           r"TPR at 1\%% FPR (\texttt{n/a} where a set has $<$100 negatives, so 1\%% FPR is "
           r"unreachable).}" % title
           + "\n" + r"\label{tab:app-det-%s}" % slug.replace("-", ""))
    L = [r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{ll" + "c" * 2 + "}", r"\toprule",
         r"Probe & Evaluation set & AUROC [95\% CI] & TPR@1\%FPR \\", r"\midrule"]
    for pr in probes:
        for k, ev in enumerate(evals):
            r = by.get((pr, ev))
            if r is None:
                continue
            name = esc(pr) if k == 0 else ""
            L.append(f"{name} & {esc(ev)} & {esc(r['AUROC'])} & {esc(r['TPR@1%FPR'])} \\\\")
        L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", cap]
    return "\n".join(L)


def steering_table(slug: str, title: str) -> str:
    p = EXPORT / f"{slug}_steering_full.json"
    if not p.is_file():
        return f"\n\\noindent\\textcolor{{red}}{{[{esc(p.name)} not found --- re-run the pipeline]}}\n"
    d = json.loads(p.read_text())

    def empty(vals):
        return all(x is None or (isinstance(x, float) and x != x) for x in vals)

    out = []
    for evalset, methods in d.items():
        first = next(iter(methods.values()))
        coefs = first["coef"]
        hdr = " & ".join(f"{c:+g}" for c in coefs)
        # A judge arm skipped for this whole set (all-NaN rows) is omitted, not dashed.
        skipped = all(empty(v.get(k) or [float("nan")])
                      for v in methods.values() for k in ("judge_beh", "judge_coh"))
        cap = (r"\caption{%s, %s: steering response. Readout NIE, judge behaviour and "
               r"judge coherence (1--10)." % (title, esc(evalset)))
        if skipped:
            cap += (r" Judge rows are omitted: the judge arm is skipped for this set, whose "
                    r"generation prompts are byte-identical under greedy decoding.")
        cap += "}"
        cap += "\n" + r"\label{tab:app-steer-%s-%s}" % (
            slug.replace("-", ""), re.sub(r"\W", "", evalset))
        L = [r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{ll" + "r" * len(coefs) + "}", r"\toprule",
             r"Direction & & " + hdr + r" \\", r"\midrule"]
        for m, v in methods.items():
            name_pending = esc(m)
            for lbl, key, fmt in (("NIE", "nie", "{:+.2f}"),
                                  ("judge beh.", "judge_beh", "{:.2f}"),
                                  ("judge coh.", "judge_coh", "{:.2f}")):
                vals = v.get(key) or []
                if empty(vals):
                    continue
                cells = [fmt.format(x) for x in vals]
                L.append(f"{name_pending} & {lbl} & " + " & ".join(cells) + r" \\")
                name_pending = ""
            L.append(r"\addlinespace[2pt]")
        L += [r"\bottomrule", r"\end{tabular}", cap]
        out.append("\n".join(L))
    return out


def pair_up(blocks: list[str]) -> list[str]:
    """Wrap table blocks two-per-float in side-by-side minipages.

    Each minipage carries its own \\caption, so the two halves are numbered as separate
    tables; an odd trailing block is emitted on its own row at the same width.
    """
    floats = []
    for i in range(0, len(blocks), 2):
        chunk = blocks[i:i + 2]
        L = [r"\begin{table}[H]", r"\centering"]
        for k, b in enumerate(chunk):
            L += [r"\begin{minipage}[t]{0.49\textwidth}", r"\centering", b, r"\end{minipage}"]
            if k == 0 and len(chunk) == 2:
                L.append(r"\hfill")
        L.append(r"\end{table}")
        floats.append("\n".join(L))
    return floats


HEADER = ["% AUTO-GENERATED by make_appendix_tables.py -- do not edit by hand.",
          "% Regenerate after any pipeline re-run so these cannot drift from exported_probes/.", ""]
parts = list(HEADER)
parts.append(r"\section{Full Detection Results}\label{app:det-tables}")
parts.append("Every probe cell against every evaluation set, as exported to "
             r"\texttt{exported\_probes/*\_detection.csv}. The out-of-distribution rows are the "
             "ones the read-side claim rests on; in-distribution AUROC is near-ceiling for two of "
             "the three behaviours and correspondingly uninformative.")
parts += pair_up([detection_table(slug, title) for slug, title in BEHAVIOURS])

OUT_DET = OUT.parent / "appendix_detection_tables.tex"
OUT_DET.write_text("\n\n".join(parts) + "\n")
print(f"wrote {OUT_DET.name}")
parts = list(HEADER)
parts.append(r"\section{Full Steering Response Curves}\label{app:steer-tables}")
parts.append("All seven coefficients for every direction and evaluation set, as exported to "
             r"\texttt{exported\_probes/*\_steering\_full.json}. Three things are easiest to see "
             "here rather than in the body: whether a curve is monotone in the coefficient, "
             "whether the negative half behaves as a control, and where judge coherence falls "
             "away from its unsteered baseline.")
_steer = [b for slug, title in BEHAVIOURS for b in steering_table(slug, title)]
parts += pair_up(_steer)

OUT_STEER = OUT.parent / "appendix_steering_tables.tex"
OUT_STEER.write_text("\n\n".join(parts) + "\n")
print(f"wrote {OUT_STEER.name}")
