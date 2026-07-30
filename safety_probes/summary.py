"""Verdict helpers + final-numbers persistence (originally notebook §4).

`auroc_pivot` and `nie_at` are the two §4 helpers, unchanged from the audited notebook. Since
plan #45 (2026-07-30) the verdicts are computed inline in each behaviour notebook — a probing
verdict (best reader per eval set, LR-family win share) at the end of its probing section and a
steering verdict (best writer by NIE at coef +1, MM win share) at the end of its steering section;
`summary.ipynb` is gone. The save/load helpers persist each behaviour's final numbers (detection
AUROC frame, steering NIE-at-coef-1) as small CSVs in `exported_probes/` for thesis tables.
"""

from __future__ import annotations

import pandas as pd

from . import runtime


def auroc_pivot(df: pd.DataFrame, behaviour: str) -> pd.DataFrame:
    """Pivot an `evaluate_grid` frame to probe × eval AUROC values, prefixed with the behaviour."""
    out = df.copy()
    out["AUROC_val"] = out["AUROC"].str.split(" ").str[0].astype(float)  # "0.923 [0.90,0.94]" -> 0.923
    out = out.pivot(index="probe", columns="eval", values="AUROC_val")
    out.columns = [f"{behaviour}: {c}" for c in out.columns]
    return out


def nie_at(results: dict, coef: float = 1.0) -> dict:
    """NIE at the calibrated coefficient +1 (= push negatives one full class gap)."""
    return {m: r["nie"][r["coef"].index(coef)] for m, r in results.items()}


# --- persistence helpers so §4 survives the per-notebook kernel split --------------------

def _summary_dir():
    d = runtime.PROJECT_ROOT / "exported_probes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_detection(grid_df: pd.DataFrame, behaviour: str) -> None:
    """Persist a behaviour's detection AUROC frame for the summary notebook."""
    grid_df.to_csv(_summary_dir() / f"{behaviour}_detection.csv", index=False)


def save_steering(nie_by_evalset: dict[str, dict], behaviour: str) -> None:
    """Persist a behaviour's NIE-at-coef-1 per eval set. `nie_by_evalset` maps a column label to a
    `steering_nie_llm_evaluation` results dict, e.g. {"ID": dec_id_results, "AI Liar (OOD)": ...}."""
    rows = {f"{behaviour}: {label}": nie_at(res) for label, res in nie_by_evalset.items()}
    pd.DataFrame(rows).to_csv(_summary_dir() / f"{behaviour}_steering.csv")


def load_detection(behaviour: str) -> pd.DataFrame:
    return pd.read_csv(_summary_dir() / f"{behaviour}_detection.csv")


def load_steering(behaviour: str) -> pd.DataFrame:
    return pd.read_csv(_summary_dir() / f"{behaviour}_steering.csv", index_col=0)


def save_steering_full(results_by_evalset: dict[str, dict], behaviour: str) -> None:
    """Persist the COMPLETE dose-response results — every coef's NIE + judge behaviour/coherence,
    per direction per eval set — for the cross-behaviour comparison notebook (plan #48), which runs
    in its own kernel with no GPU. NaN (judge-skipped points, e.g. deception ID post-#46) survives
    the round-trip: Python's json emits/accepts the NaN literal."""
    import json
    payload = {label: {method: {k: [float(x) for x in v] for k, v in r.items()}
                       for method, r in res.items()}
               for label, res in results_by_evalset.items()}
    (_summary_dir() / f"{behaviour}_steering_full.json").write_text(json.dumps(payload, indent=1))


def load_steering_full(behaviour: str) -> dict:
    """{eval_set_label: {method: {"coef": [...], "nie": [...], "judge_beh": [...], "judge_coh": [...]}}}"""
    import json
    return json.loads((_summary_dir() / f"{behaviour}_steering_full.json").read_text())
