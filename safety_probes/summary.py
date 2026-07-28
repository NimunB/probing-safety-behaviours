"""Cross-behaviour summary — the headline read/write verdict (notebook §4).

Reproduces §4 across the two per-behaviour notebooks, which run in separate kernels. Each notebook
persists two small CSVs to `exported_probes/` (detection AUROC frame, steering NIE-at-coef-1); this
module + `summary.ipynb` load them and compute the verdict. `auroc_pivot` and `nie_at` are the two
§4 helpers, unchanged from the notebook.
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
