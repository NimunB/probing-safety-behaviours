"""Probe export/import — the ~0.5 MB artefact that outlives the run (notebook §1h, plan #43).

Exports NO activations. A fitted probe is a handful of vectors; activations were only ever its
training data. Live detection and steering both run a forward pass on the user's own input, so
nothing needs to be precomputed or stored.

WHAT A BUNDLE CONTAINS
----------------------
  detection : the 5 probes at the detection layer, as state_dicts
  steering  : CALIBRATED directions, per method per block layer
  layers    : detect_layer (chosen by pair-aware CV AUROC) + steer_block (chosen by NIE sweep)
  provenance: model_name, d_model, seed, format_version, timestamp

`steer_dirs` MUST be the output of `calibrate_block()`, never `build_directions()`.
`calibrate()` bakes the [GoT §6.1] mean-gap scale into the vector itself, so coefficient 1.0
equals one cross-class gap with no separate scalar to carry. Exporting raw directions would
silently make any downstream strength control meaningless.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import torch as t

from . import runtime
from .probes import AttentionProbe, LRProbe, MMProbe

FORMAT_VERSION = 1


def export_dir() -> Path:
    d = runtime.PROJECT_ROOT / "exported_probes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_behaviour(spec, grid, detect_layer, steer_block, steer_dirs, d_model, notes=""):
    """Serialize everything needed to RUN one behaviour's probes, without re-running the pipeline.

    All tensors are cast to fp32 on CPU: probes are tiny, and fp16 directions would reintroduce
    the dtype mismatch `MMProbe.score` had to work around (plan #21).
    """
    detection = {}
    for name, (probe, pooling) in grid.items():
        if isinstance(probe, list):  # LR-attention: a list of per-seed probes
            detection[name] = {
                "kind": "attention_seeds",
                "pooling": pooling,
                "n_seeds": len(probe),
                "state_dicts": [{k: v.cpu().float() for k, v in p.state_dict().items()}
                                for p in probe],
            }
        else:
            detection[name] = {
                "kind": type(probe).__name__,
                "pooling": pooling,
                "state_dict": {k: v.cpu().float() for k, v in probe.state_dict().items()},
            }

    steering = {method: {int(layer): vec.cpu().float() for layer, vec in layers.items()}
                for method, layers in steer_dirs.items()}

    bundle = {
        # provenance — lets a loader REFUSE a mismatched bundle instead of silently misbehaving,
        # the same failure mode the content-blind activation cache has.
        "format_version": FORMAT_VERSION,
        "model_name": runtime.MODEL_NAME,
        "d_model": int(d_model),
        "seed": runtime.SEED,
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # behaviour identity (drives the UI dropdown + heatmap legend)
        "behaviour": spec.name,
        "positive_name": spec.positive_name,
        "negative_name": spec.negative_name,
        "mask_policy": spec.mask_policy,
        "readout_targets": list(spec.readout_targets),
        # the two layer choices, made by two DIFFERENT criteria
        "detect_layer": int(detect_layer),               # pair-aware 5-fold CV AUROC
        "steer_block": [int(l) for l in steer_block],    # NIE sweep
        "steer_coefs": list(spec.steer_coefs),
        # the payload
        "detection": detection,
        "steering": steering,
        "notes": notes,
    }

    path = export_dir() / f"{spec.name}_probes.pt"
    t.save(bundle, path)
    print(f"Exported {spec.name} -> {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"  detection layer {detect_layer}: {list(detection)}")
    print(f"  steering block  {steer_block}: {list(steering)}")
    return path


def load_behaviour(name_or_path, expect_model=None, expect_d_model=None):
    """Load a bundle, refusing one built for a different model / width.

    The refusal is deliberate insurance against the failure mode the activation cache has:
    silently serving artefacts that don't match the current setup.
    """
    path = Path(name_or_path)
    if not path.exists():
        path = export_dir() / f"{name_or_path}_probes.pt"
    b = t.load(path, weights_only=False)
    if expect_model and b["model_name"] != expect_model:
        raise ValueError(f"{path.name} was built for {b['model_name']}, not {expect_model}")
    if expect_d_model and b["d_model"] != expect_d_model:
        raise ValueError(f"{path.name} has d_model={b['d_model']}, not {expect_d_model}")
    return b


def rebuild_detection_probe(bundle, probe_name):
    """Reconstruct a live probe object from an exported state_dict."""
    entry, d = bundle["detection"][probe_name], bundle["d_model"]
    if entry["kind"] == "MMProbe":
        return MMProbe(entry["state_dict"]["direction"])
    if entry["kind"] == "LRProbe":
        sd = entry["state_dict"]
        p = LRProbe(d, scaler_mean=sd.get("scaler_mean"), scaler_scale=sd.get("scaler_scale"))
        p.load_state_dict(sd)
        return p.eval()
    if entry["kind"] == "attention_seeds":
        seeds = [AttentionProbe(d_model=d) for _ in entry["state_dicts"]]
        for probe, sd in zip(seeds, entry["state_dicts"]):
            probe.load_state_dict(sd)
            probe.eval()
        return seeds
    raise ValueError(f"unknown probe kind {entry['kind']}")
