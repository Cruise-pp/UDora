#!/usr/bin/env python3
"""
Consistency experiment runner for InjecAgent (ds, first 10 train cases).

Runs the same setting 5 times (without fixed seed), saves each run's raw UDora
results, and writes aggregate consistency statistics.
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InjecAgent consistency experiment on first 10 ds train cases."
    )

    # Keep defaults aligned with the user's target setting.
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--task", type=str, default="ds", choices=["ds"])
    parser.add_argument("--mode", type=str, default="train", choices=["train"])
    parser.add_argument("--num_location", type=int, default=3)
    parser.add_argument("--add_space_before_target", action="store_true", default=True)
    parser.add_argument("--early_stop", action="store_true", default=True)

    # Keep remaining attack args configurable for flexibility.
    parser.add_argument("--optim_str_init", type=str, default="x x x x x x x x x x x x x x x x x x x x x x x x x")
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--search_width", type=int, default=256)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--n_replace", type=int, default=1)
    parser.add_argument("--prefix_update_frequency", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--buffer_size", type=int, default=0)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--use_mellowmax", action="store_true")
    parser.add_argument("--readable", action="store_true")
    parser.add_argument("--verbosity", type=str, default="INFO")

    # Consistency experiment controls.
    parser.add_argument("--num_runs", type=int, default=5, help="How many repeated runs to execute.")
    parser.add_argument("--num_cases", type=int, default=10, help="How many leading train cases to evaluate.")
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="results/consistency/injecagent_ds_first10",
        help="Root directory for all repeated runs and summary files.",
    )
    parser.add_argument(
        "--resume_existing",
        action="store_true",
        help="Resume incomplete runs and skip completed runs if outputs already exist.",
    )

    return parser.parse_args()


def bool_from_udora_flag(value: Any) -> bool:
    """Normalize UDora bool/list bool flag to a plain bool."""
    if isinstance(value, list):
        return bool(value[0]) if value else False
    return bool(value)


def load_run_results(output_path: str) -> List[Any]:
    with open(output_path, "rb") as f:
        return pickle.load(f)


def safe_load_existing_count(output_path: str) -> int:
    if not os.path.exists(output_path):
        return 0
    try:
        return len(load_run_results(output_path))
    except Exception:
        return 0


def summarize_single_run(results: List[Any]) -> Dict[str, Any]:
    """Compute per-run summary metrics."""
    best_success = [bool_from_udora_flag(res.best_success) for res in results]
    last_success = [bool_from_udora_flag(res.last_success) for res in results]
    final_success_or = [b or l for b, l in zip(best_success, last_success)]

    return {
        "num_cases": len(results),
        "asr_best": mean(best_success) if best_success else 0.0,
        "asr_last": mean(last_success) if last_success else 0.0,
        "asr_final_or": mean(final_success_or) if final_success_or else 0.0,
        "best_success_by_case": best_success,
        "last_success_by_case": last_success,
        "final_success_or_by_case": final_success_or,
        "best_strings_by_case": [res.best_string for res in results],
        "last_strings_by_case": [res.last_string for res in results],
        "best_losses_by_case": [float(res.best_loss) for res in results],
    }


def jaccard(a: List[bool], b: List[bool]) -> float:
    """Jaccard of successful-case sets represented by bool lists."""
    aset = {i for i, v in enumerate(a) if v}
    bset = {i for i, v in enumerate(b) if v}
    union = aset | bset
    if not union:
        return 1.0
    return len(aset & bset) / len(union)


def pairwise_jaccard(best_success_lists: List[List[bool]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for i in range(len(best_success_lists)):
        for j in range(i + 1, len(best_success_lists)):
            pairs.append(
                {
                    "run_i": i + 1,
                    "run_j": j + 1,
                    "jaccard_success_set": jaccard(best_success_lists[i], best_success_lists[j]),
                }
            )
    return pairs


def aggregate(run_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not run_summaries:
        return {}

    num_runs = len(run_summaries)
    num_cases = run_summaries[0]["num_cases"]

    asr_best_list = [r["asr_best"] for r in run_summaries]
    asr_last_list = [r["asr_last"] for r in run_summaries]
    asr_final_or_list = [r["asr_final_or"] for r in run_summaries]
    best_success_matrix = [r["best_success_by_case"] for r in run_summaries]
    last_success_matrix = [r["last_success_by_case"] for r in run_summaries]
    final_or_success_matrix = [r["final_success_or_by_case"] for r in run_summaries]
    best_string_matrix = [r["best_strings_by_case"] for r in run_summaries]

    per_case_best_success_count = [
        sum(best_success_matrix[r][c] for r in range(num_runs)) for c in range(num_cases)
    ]
    per_case_last_success_count = [
        sum(last_success_matrix[r][c] for r in range(num_runs)) for c in range(num_cases)
    ]
    per_case_best_string_unique_count = [
        len({best_string_matrix[r][c] for r in range(num_runs)}) for c in range(num_cases)
    ]
    per_case_final_or_success_count = [
        sum(final_or_success_matrix[r][c] for r in range(num_runs)) for c in range(num_cases)
    ]

    return {
        "overall": {
            "num_runs": num_runs,
            "num_cases": num_cases,
            "mean_asr_best": mean(asr_best_list),
            "std_asr_best": pstdev(asr_best_list),
            "mean_asr_last": mean(asr_last_list),
            "std_asr_last": pstdev(asr_last_list),
            "mean_asr_final_or": mean(asr_final_or_list),
            "std_asr_final_or": pstdev(asr_final_or_list),
        },
        "per_case": [
            {
                "case_index": c,
                "best_success_count_out_of_runs": per_case_best_success_count[c],
                "last_success_count_out_of_runs": per_case_last_success_count[c],
                "final_or_success_count_out_of_runs": per_case_final_or_success_count[c],
                "best_string_unique_count": per_case_best_string_unique_count[c],
            }
            for c in range(num_cases)
        ],
        "pairwise_success_set_jaccard": pairwise_jaccard(best_success_matrix),
        "pairwise_success_set_jaccard_final_or": pairwise_jaccard(final_or_success_matrix),
    }


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_once(
    run_idx: int,
    args: argparse.Namespace,
    model_name: str,
    model: Any,
    tokenizer: Any,
    goals_10: List[str],
    targets_10: List[str],
    sanitize_filename_fn: Any,
    create_udora_config_fn: Any,
    run_injecagent_attack_fn: Any,
    num_cases_expected: int,
) -> Tuple[str, Dict[str, Any]]:
    run_dir = os.path.join(args.base_output_dir, f"run_{run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    filename = (
        f"{sanitize_filename_fn(args.model_id)}_injecagent_{args.task}_"
        f"steps{args.num_steps}_width{args.search_width}_topk{args.topk}_"
        f"replace{args.n_replace}_locations{args.num_location}_weight{args.weight}_"
        f"{'sequential' if args.sequential else 'joint'}_"
        f"{'readable' if args.readable else 'standard'}_{args.mode}.pkl"
    )
    output_path = os.path.join(run_dir, filename)

    config = create_udora_config_fn(args, model_name, args.task)

    # Important: no fixed seed for consistency-variance experiment.
    config.seed = None

    existing_count = safe_load_existing_count(output_path)

    print(f"\n=== Run {run_idx}/{args.num_runs} ===")
    print(f"Output path: {output_path}")

    if args.resume_existing and existing_count >= num_cases_expected:
        print(f"Run {run_idx}: detected completed output ({existing_count}/{num_cases_expected}), skipping.")
    else:
        should_resume = args.resume_existing and (0 < existing_count < num_cases_expected)
        if should_resume:
            print(f"Run {run_idx}: resuming from partial progress ({existing_count}/{num_cases_expected}).")
        elif existing_count > 0 and not args.resume_existing:
            print(f"Run {run_idx}: restarting despite existing output ({existing_count}/{num_cases_expected}).")

        run_injecagent_attack_fn(
            model=model,
            tokenizer=tokenizer,
            goals=goals_10,
            targets=targets_10,
            config=config,
            output_path=output_path,
            resume=should_resume,
        )

    results = load_run_results(output_path)
    run_summary = summarize_single_run(results)
    run_summary["run_index"] = run_idx
    run_summary["output_path"] = output_path
    return output_path, run_summary


def main() -> None:
    args = parse_args()
    # Lazy imports allow --help to run even before heavy dependencies are installed.
    from attack_utils import get_model_name, load_model_and_tokenizer, sanitize_filename
    from injecagent import create_udora_config, get_data_path, prepare_data, run_injecagent_attack

    model_name = get_model_name(args.model_id)
    data_path = get_data_path(model_name, args.mode, args.task)
    print(f"Model name: {model_name}")
    print(f"Data path: {data_path}")

    goals, targets = prepare_data(data_path, args.mode)
    num_cases = min(args.num_cases, len(goals))
    goals_10 = goals[:num_cases]
    targets_10 = targets[:num_cases]
    print(f"Using first {num_cases} cases out of {len(goals)}")

    # Load model once and reuse across repeated runs.
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    run_summaries: List[Dict[str, Any]] = []
    run_outputs: List[str] = []

    for run_idx in range(1, args.num_runs + 1):
        out_path, run_summary = run_once(
            run_idx=run_idx,
            args=args,
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            goals_10=goals_10,
            targets_10=targets_10,
            sanitize_filename_fn=sanitize_filename,
            create_udora_config_fn=create_udora_config,
            run_injecagent_attack_fn=run_injecagent_attack,
            num_cases_expected=num_cases,
        )
        run_outputs.append(out_path)
        run_summaries.append(run_summary)

    aggregate_summary = aggregate(run_summaries)

    experiment_meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "setting": {
            "model_id": args.model_id,
            "task": args.task,
            "mode": args.mode,
            "add_space_before_target": args.add_space_before_target,
            "num_location": args.num_location,
            "early_stop": args.early_stop,
            "num_steps": args.num_steps,
            "search_width": args.search_width,
            "topk": args.topk,
            "n_replace": args.n_replace,
            "weight": args.weight,
            "sequential": args.sequential,
            "readable": args.readable,
            "num_runs": args.num_runs,
            "num_cases": num_cases,
            "seed": None,
        },
        "run_outputs": run_outputs,
        "runs": run_summaries,
        "aggregate": aggregate_summary,
    }

    summary_json = os.path.join(args.base_output_dir, "summary.json")
    write_json(summary_json, experiment_meta)

    print("\n=== Consistency Experiment Complete ===")
    print(f"Summary JSON: {summary_json}")
    print(
        "mean_asr_best={:.4f}, std_asr_best={:.4f}".format(
            aggregate_summary["overall"]["mean_asr_best"],
            aggregate_summary["overall"]["std_asr_best"],
        )
    )


if __name__ == "__main__":
    main()
