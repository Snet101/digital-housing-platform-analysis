"""Repeat synthetic assumptions without selecting a winning test configuration."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from housing_model import run, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "status.json", {"state": "incomplete"})
    results = []
    try:
        for seed in [41, 42, 43]:
            for curvature in [0.0, 10.0, 20.0]:
                metrics = run(args.input, args.out / f"seed-{seed}-curve-{curvature:g}", seed, curvature)
                results.append({"seed": seed, "curvature": curvature, "metrics": metrics})
        save_json(args.out / "sensitivity.json", results)
        save_json(args.out / "status.json", {"state": "complete"})
    except Exception:
        save_json(args.out / "status.json", {"state": "failed"})
        raise


if __name__ == "__main__":
    main()
