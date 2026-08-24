from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from ..paths import PROCESSED_DATA_ROOT


THEMES = ("natural_elements", "traffic_elements", "built_elements")
DIMENSIONS = ("Beautiful", "Depressing", "Lively", "Safe", "Wealthy", "Boring")
RESAMPLES = 30_000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def bootstrap_mean(values: list[float], seed: int) -> tuple[float, list[float]]:
    data = np.asarray(values, dtype=float)
    random = np.random.default_rng(seed)
    draws = data[random.integers(0, data.size, size=(RESAMPLES, data.size))].mean(axis=1)
    return float(data.mean()), [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def exact_signflip_p(values: list[float]) -> float:
    data = np.asarray(values, dtype=float)
    observed = abs(float(data.mean()))
    count = sum(
        abs(float((data * np.asarray(signs)).mean())) >= observed - 1e-12
        for signs in itertools.product((-1.0, 1.0), repeat=data.size)
    )
    return count / (2**data.size)


def recompute(data_dir: Path) -> dict:
    behavior = read_csv(data_dir / "perception_7b_behavior_matrix.csv")
    patch = read_csv(data_dir / "perception_7b_state_replacement_matrix.csv")
    if len(behavior) != 216 or len(patch) != 216:
        raise ValueError("Expected 216 rows in each released matrix")
    scenes = sorted({row["scene_id"] for row in behavior})
    keys = [(theme, dimension) for theme in THEMES for dimension in DIMENSIONS]
    reference = np.asarray(
        [
            next(
                float(row["reference_effect"])
                for row in behavior
                if (row["theme"], row["dimension"]) == key
            )
            for key in keys
        ]
    )
    matrix = np.asarray(
        [
            [
                next(
                    float(row["signed_theme_score"])
                    for row in behavior
                    if row["scene_id"] == scene
                    and (row["theme"], row["dimension"]) == key
                )
                for key in keys
            ]
            for scene in scenes
        ]
    )
    correlation = float(np.corrcoef(matrix.mean(axis=0), reference)[0, 1])
    random = np.random.default_rng(20260818)
    sampled = matrix[
        random.integers(0, len(scenes), size=(RESAMPLES, len(scenes)))
    ].mean(axis=1)
    centered = sampled - sampled.mean(axis=1, keepdims=True)
    centered_reference = reference - reference.mean()
    draws = (centered @ centered_reference) / np.sqrt(
        np.square(centered).sum(axis=1) * np.square(centered_reference).sum()
    )
    draws = draws[np.isfinite(draws)]
    correlation_ci = [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]
    denominator = float(np.abs(reference).sum())
    scene_estimands = []
    for scene in scenes:
        rows = [row for row in patch if row["scene_id"] == scene]
        target = sum(
            float(row["reference_effect"]) * float(row["target_change"])
            for row in rows
        ) / denominator
        random_effect = sum(
            float(row["reference_effect"]) * float(row["random_change"])
            for row in rows
        ) / denominator
        scene_estimands.append(
            {
                "target": target,
                "random": random_effect,
                "specificity": target - abs(random_effect),
            }
        )
    target_values = [row["target"] for row in scene_estimands]
    specificity_values = [row["specificity"] for row in scene_estimands]
    target_estimate, target_ci = bootstrap_mean(target_values, 20260908)
    specificity_estimate, specificity_ci = bootstrap_mean(
        specificity_values, 20260909
    )
    return {
        "clean_profile_reference_correlation": {
            "n_scenes": len(scenes),
            "estimate": correlation,
            "ci95": correlation_ci,
            "bootstrap_resamples": RESAMPLES,
            "bootstrap_seed": 20260818,
        },
        "target_profile_attenuation": {
            "n_scenes": len(scenes),
            "estimate": target_estimate,
            "ci95": target_ci,
            "bootstrap_resamples": RESAMPLES,
            "bootstrap_seed": 20260908,
            "exact_two_sided_signflip_p": exact_signflip_p(target_values),
        },
        "target_minus_absolute_random_specificity": {
            "n_scenes": len(scenes),
            "estimate": specificity_estimate,
            "ci95": specificity_ci,
            "bootstrap_resamples": RESAMPLES,
            "bootstrap_seed": 20260909,
            "exact_two_sided_signflip_p": exact_signflip_p(specificity_values),
        },
        "registered_and_usable_pair_units": {
            "behavior": [216, 216],
            "state_replacement": [216, 216],
        },
        "equal_norm_checks_passed": sum(
            str(row["equal_norm_check_passed"]).lower() == "true" for row in patch
        ),
    }


def main() -> int:
    default = PROCESSED_DATA_ROOT / "model_outputs"
    parser = argparse.ArgumentParser(
        description="Recompute released 7B Perception state-replacement statistics."
    )
    parser.add_argument("--data-dir", type=Path, default=default)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = recompute(args.data_dir)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

