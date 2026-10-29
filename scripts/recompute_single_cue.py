"""Recompute five reported single-cue effects from the released scene scores.

Run from the repository root:
    python -B scripts/recompute_single_cue.py
Requires Python 3.10+ and NumPy. No model inference is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def recompute_rows(scores, stored_estimands, stored_summary, config, figure_rows):
    scenes = sorted({row["scene_id"] for row in scores})
    if len(scenes) != 12 or len(scores) != 60:
        raise ValueError("Expected 12 scenes and 60 scored images")
    if scenes != sorted(config["scene_ids"]):
        raise ValueError("Scene identifiers differ from the analysis configuration")
    expected_conditions = {"baseline", "sham", "tree_canopy", "visible_activity", "facade_repair"}
    indexed = {}
    for row in scores:
        key = (row["scene_id"], row["condition"])
        if key in indexed:
            raise ValueError("Duplicate scene/condition")
        if row["condition"] not in expected_conditions:
            raise ValueError("Unknown condition")
        values = {dim: float(row[dim]) for dim in config["dimensions"]}
        if any(not np.isfinite(value) or not 0 <= value <= 10 for value in values.values()):
            raise ValueError("Scores must be finite and between 0 and 10")
        indexed[key] = values
    if set(indexed) != {(scene, cond) for scene in scenes for cond in expected_conditions}:
        raise ValueError("Incomplete scene-condition design")
    stored = {(r["scene_id"], r["condition"], r["dimension"]): r for r in stored_estimands}
    if len(stored) != 60 or len(stored_estimands) != 60:
        raise ValueError("Expected 60 unique prespecified scene-level effects")
    expected = {(r["condition"], r["dimension"]): r for r in stored_summary}
    source = {(r["condition"], r["dimension"]): r for r in figure_rows if r["panel"] == "d"}
    if len(expected) != 5 or len(source) != 5:
        raise ValueError("Expected five summary and figure-source effects")
    rng = np.random.default_rng(config["bootstrap_seed"])
    summaries = []
    seen = set()
    for link in config["primary_links"]:
        condition, dimension = link["condition"], link["dimension"]
        vals = []
        for scene in scenes:
            base = indexed[(scene, "baseline")][dimension]
            control = indexed[(scene, "sham")][dimension]
            target = indexed[(scene, condition)][dimension]
            target_change, control_change = target - base, control - base
            effect = target_change - control_change
            key = (scene, condition, dimension)
            seen.add(key)
            row = stored[key]
            if row["direction"] != link["direction"]:
                raise ValueError("Prespecified direction mismatch")
            for column, observed in (("target_change", target_change), ("sham_change", control_change), ("estimand", effect)):
                if not np.isclose(observed, float(row[column]), rtol=0, atol=1e-12):
                    raise ValueError("Scene estimate mismatch: " + column)
            vals.append(effect)
        vals = np.asarray(vals, dtype=float)
        draws = vals[rng.integers(0, len(vals), size=(config["bootstrap_resamples"], len(vals)))].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        result = {**link, "n_scenes": len(vals), "estimate": float(vals.mean()),
                  "ci95_low": float(low), "ci95_high": float(high)}
        for column in ("estimate", "ci95_low", "ci95_high"):
            if not np.isclose(result[column], float(expected[(condition, dimension)][column]), rtol=0, atol=1e-12):
                raise ValueError("Published summary mismatch: " + column)
        for calculated, source_column in (("estimate", "estimate"), ("ci95_low", "ci_low"), ("ci95_high", "ci_high")):
            if not np.isclose(result[calculated], float(source[(condition, dimension)][source_column]), rtol=0, atol=1e-12):
                raise ValueError("Figure S9d source mismatch: " + calculated)
        summaries.append(result)
    if seen != set(stored):
        raise ValueError("Unexpected scene-level effects")
    return {"status": "pass", "independent_scenes": 12, "scored_images": 60,
            "scene_effects_checked": 60, "bootstrap_resamples": config["bootstrap_resamples"],
            "bootstrap_seed": config["bootstrap_seed"],
            "interval": "Pointwise percentile 95%; no multiplicity adjustment",
            "matches_released_summary_and_FigS9d": True, "effects": summaries}


def recompute(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "input_manifest.json").read_text(encoding="utf-8"))
    inputs = ["perception_single_cue_scores.csv",
              "perception_single_cue_scene_estimands.csv",
              "perception_single_cue_summary.csv",
              "single_cue_analysis_config.json", "figure_s9_source.csv"]
    hashes = {row["path"]: row["sha256"] for row in manifest["inputs"]}
    if set(hashes) != set(inputs) | {"perception_single_cue_qc.csv"}:
        raise ValueError("Unexpected input manifest")
    for name, expected_hash in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Input checksum mismatch: " + name)
    config = json.loads((root / inputs[3]).read_text(encoding="utf-8"))
    return recompute_rows(csv_rows(root / inputs[0]), csv_rows(root / inputs[1]),
                          csv_rows(root / inputs[2]), config, csv_rows(root / inputs[4]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/processed/single_cue")
    args = parser.parse_args()
    print(json.dumps(recompute(args.data_root), indent=2))
