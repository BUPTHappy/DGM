#!/usr/bin/env python3
"""Run eval_sim.py many times with different seeds and aggregate scores.

Example (dgm_pusht, 50 model-RNG seeds):
  python scripts/eval/multi_seed_eval.py \\
    --checkpoint checkpoints/pusht_dgm.ckpt \\
    --output_dir eval_out/pusht_dgm_multiseed \\
    --num_runs 50 \\
    --seed_mode model \\
    --base_seed 0 \\
    --num_sampling_steps 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed inference for a trained checkpoint via eval_sim.py"
    )
    p.add_argument("-c", "--checkpoint", required=True, help="Path to .ckpt")
    p.add_argument("-o", "--output_dir", required=True, help="Directory for all runs + summary")
    p.add_argument("-d", "--device", default="cuda:0")
    p.add_argument("--num_runs", type=int, default=50, help="Number of evaluation runs")
    p.add_argument(
        "--seed_mode",
        choices=["model", "env", "both"],
        default="model",
        help=(
            "model: vary RNG seed (sampling noise), keep env ICs fixed; "
            "env: vary test_start_seed (different env initial conditions); "
            "both: vary both per run"
        ),
    )
    p.add_argument("--base_seed", type=int, default=0, help="First model RNG seed")
    p.add_argument(
        "--base_test_start_seed",
        type=int,
        default=100000,
        help="First env test_start_seed (used in env/both modes)",
    )
    p.add_argument(
        "--env_seed_stride",
        type=int,
        default=1000,
        help="Increment of test_start_seed between runs in env/both modes",
    )
    p.add_argument("--n_test", type=int, default=None, help="Override n_test per run")
    p.add_argument("--num_sampling_steps", type=int, default=None)
    p.add_argument("--stochasticity_rate", type=float, default=None)
    p.add_argument("--window_size", type=int, default=None)
    p.add_argument("--lambda_local", type=float, default=None)
    p.add_argument("--use_ucgm", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Libero dataset path override",
    )
    p.add_argument(
        "--start_run",
        type=int,
        default=0,
        help="Resume from this run index (0-based)",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a run if its eval_log json already exists",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )
    return p.parse_args()


def extract_score(eval_results: Dict[str, Any]) -> Optional[float]:
    preferred = [
        "test/mean_score",
        "test_mean_score",
        "test/pusht_mean_score",
        "test/libero10_mean_score",
    ]
    for key in preferred:
        if key in eval_results:
            return float(eval_results[key])

    score_keys = [k for k in eval_results if "mean_score" in k and "test" in k]
    if score_keys:
        return float(eval_results[score_keys[0]])
    return None


def build_cmd(
    args: argparse.Namespace,
    run_output_dir: Path,
    model_seed: Optional[int],
    test_start_seed: Optional[int],
) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval_sim.py"),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--output_dir",
        str(run_output_dir),
        "--device",
        args.device,
    ]
    if model_seed is not None:
        cmd.extend(["--seed", str(model_seed)])
    if test_start_seed is not None:
        cmd.extend(["--test_start_seed", str(test_start_seed)])
    if args.n_test is not None:
        cmd.extend(["--n_test", str(args.n_test)])
    if args.num_sampling_steps is not None:
        cmd.extend(["--num_sampling_steps", str(args.num_sampling_steps)])
    if args.stochasticity_rate is not None:
        cmd.extend(["--stochasticity_rate", str(args.stochasticity_rate)])
    if args.window_size is not None:
        cmd.extend(["--window_size", str(args.window_size)])
    if args.lambda_local is not None:
        cmd.extend(["--lambda_local", str(args.lambda_local)])
    if args.use_ucgm:
        cmd.append("--use_ucgm")
    if args.no_ema:
        cmd.append("--no_ema")
    if args.dataset_path:
        cmd.extend(["--dataset_path", args.dataset_path])
    return cmd


def seeds_for_run(args: argparse.Namespace, run_idx: int) -> tuple[Optional[int], Optional[int]]:
    model_seed = None
    test_start_seed = None
    if args.seed_mode in ("model", "both"):
        model_seed = args.base_seed + run_idx
    if args.seed_mode in ("env", "both"):
        test_start_seed = args.base_test_start_seed + run_idx * args.env_seed_stride
    return model_seed, test_start_seed


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = Path(args.checkpoint).name
    results: List[Dict[str, Any]] = []

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {output_dir}")
    print(f"Runs: {args.num_runs} | seed_mode={args.seed_mode}")

    for run_idx in range(args.start_run, args.num_runs):
        model_seed, test_start_seed = seeds_for_run(args, run_idx)
        run_dir = output_dir / f"run_{run_idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        eval_log_path = run_dir / f"eval_log_{ckpt_name}.json"

        record: Dict[str, Any] = {
            "run_idx": run_idx,
            "model_seed": model_seed,
            "test_start_seed": test_start_seed,
            "run_dir": str(run_dir),
            "eval_log": str(eval_log_path),
            "status": "pending",
            "score": None,
        }

        if args.skip_existing and eval_log_path.exists():
            with open(eval_log_path, "r") as f:
                eval_results = json.load(f)
            score = extract_score(eval_results)
            record["status"] = "skipped_existing"
            record["score"] = score
            results.append(record)
            print(f"[run {run_idx}] skip existing, score={score}")
            continue

        cmd = build_cmd(args, run_dir, model_seed, test_start_seed)
        print(f"\n[run {run_idx}/{args.num_runs - 1}] model_seed={model_seed} "
              f"test_start_seed={test_start_seed}")
        print(" ".join(cmd))

        if args.dry_run:
            record["status"] = "dry_run"
            results.append(record)
            continue

        env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" not in env and "cuda" in args.device:
            env["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]

        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
        if proc.returncode != 0:
            record["status"] = "failed"
            record["returncode"] = proc.returncode
            results.append(record)
            print(f"[run {run_idx}] FAILED (code={proc.returncode})")
            continue

        if not eval_log_path.exists():
            record["status"] = "missing_log"
            results.append(record)
            print(f"[run {run_idx}] missing eval log: {eval_log_path}")
            continue

        with open(eval_log_path, "r") as f:
            eval_results = json.load(f)
        score = extract_score(eval_results)
        record["status"] = "ok"
        record["score"] = score
        results.append(record)
        print(f"[run {run_idx}] score={score}")

    scores = [r["score"] for r in results if r.get("score") is not None]
    summary: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed_mode": args.seed_mode,
        "num_runs": args.num_runs,
        "base_seed": args.base_seed,
        "base_test_start_seed": args.base_test_start_seed,
        "env_seed_stride": args.env_seed_stride,
        "num_sampling_steps": args.num_sampling_steps,
        "stochasticity_rate": args.stochasticity_rate,
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "n_failed": sum(1 for r in results if r["status"] == "failed"),
        "scores": scores,
        "mean": float(np.mean(scores)) if scores else None,
        "std": float(np.std(scores)) if scores else None,
        "min": float(np.min(scores)) if scores else None,
        "max": float(np.max(scores)) if scores else None,
        "median": float(np.median(scores)) if scores else None,
        "runs": results,
    }

    summary_path = output_dir / "multi_seed_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    csv_path = output_dir / "multi_seed_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_idx", "model_seed", "test_start_seed", "status", "score"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "run_idx": r["run_idx"],
                    "model_seed": r["model_seed"],
                    "test_start_seed": r["test_start_seed"],
                    "status": r["status"],
                    "score": r["score"],
                }
            )

    print("\n========== Multi-seed summary ==========")
    print(f"OK runs: {summary['n_ok']} / {args.num_runs}")
    if scores:
        print(
            f"mean={summary['mean']:.4f}  std={summary['std']:.4f}  "
            f"min={summary['min']:.4f}  max={summary['max']:.4f}  "
            f"median={summary['median']:.4f}"
        )
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
