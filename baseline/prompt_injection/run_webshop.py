"""
Prompt Injection Baseline — WebShop Runner

Evaluates the fixed-string prompt injection baseline on the WebShop dataset.
Data loading and progress utilities mirror webshop.py for consistency.

Usage:
    # Task 0, train split
    python baseline/prompt_injection/run_webshop.py \\
        --model_id meta-llama/Llama-3.1-8B-Instruct --task 0 --mode train

    # Task 2, training split
    python baseline/prompt_injection/run_webshop.py \\
        --model_id meta-llama/Llama-3.1-8B-Instruct --task 2 --mode train
"""

import argparse
import ast
import os
import sys
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

# Allow imports from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from attack_utils import (
    get_model_name,
    sanitize_filename,
    load_model_and_tokenizer,
    load_existing_results,
    save_results_incrementally,
    print_attack_progress,
    print_final_results,
    get_success_status,
)
from baseline.prompt_injection.prompt_injection import PromptInjection


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt Injection Baseline — WebShop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=0,
        help="WebShop task index [0, 1, 2, 3]",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate per example",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results if available",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/baseline/webshop/",
        help="Directory to save attack results",
    )
    return parser.parse_args()


def get_data_path(model_name: str, mode: str) -> str:
    return f"data/webshop/{mode}_{model_name}.csv"


def generate_output_path(
    model_id: str, task: int, mode: str, output_dir: str
) -> str:
    sanitized = sanitize_filename(model_id)
    filename = f"{sanitized}_webshop_task{task}_{mode}.pkl"
    path = os.path.join(output_dir, mode)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)


def insert_optim_str(observation: str, asin: str) -> str:
    """Insert the {optim_str} placeholder after the target ASIN description."""
    parts = [p.strip() for p in observation.split("[SEP]")]
    for i, part in enumerate(parts):
        if part == asin and i + 1 < len(parts):
            parts[i + 1] = parts[i + 1] + " {optim_str}"
            break
    return " [SEP] ".join(parts)


def prepare_data(
    data_path: str, task: int, mode: str
) -> Tuple[List[str], List[str]]:
    """Load WebShop data and insert the {optim_str} placeholder."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path)

    if mode == "train":
        goals = data[f"adv_search_history_task_{task}"].tolist()
        targets = data[f"adv_target_task_{task}"].tolist()
    else:
        goals = [data[f"adv_search_history_task_{task}"].tolist()[-1]]
        targets = [data[f"adv_target_task_{task}"].tolist()[-1]]

    print(f"Loaded {len(goals)} examples for {mode} mode")
    return goals, targets


def main() -> None:
    args = parse_arguments()

    model_name = get_model_name(args.model_id)
    data_path = get_data_path(model_name, args.mode)
    output_path = generate_output_path(
        args.model_id, args.task, args.mode, args.output_dir
    )

    print(f"Model: {model_name} | Task: {args.task} | Mode: {args.mode}")
    print(f"Data:   {data_path}")
    print(f"Output: {output_path}")

    model, tokenizer = load_model_and_tokenizer(args.model_id)
    goals, targets = prepare_data(data_path, args.task, args.mode)

    attacker = PromptInjection(
        model, tokenizer, dataset="webshop", max_new_tokens=args.max_new_tokens
    )

    results = []
    success_count = 0
    start_index = 0

    if args.resume:
        results, start_index, success_count = load_existing_results(output_path)

    for i in tqdm(range(start_index, len(goals)), desc="WebShop Prompt Injection"):
        target = targets[i]

        try:
            current_message = ast.literal_eval(goals[i])
        except (ValueError, SyntaxError) as e:
            print(f"Skipping goal {i} — parse error: {e}")
            continue

        # Insert {optim_str} placeholder after the target ASIN description
        current_message[-1]["content"] = insert_optim_str(
            current_message[-1]["content"], target
        )

        result = attacker.run_single(current_message, [target])
        success = get_success_status(result)
        success_count += int(success)
        print_attack_progress(
            i, len(goals), success, success_count, start_index,
            extra_info=f"Target: {target}",
        )
        results.append(result)
        save_results_incrementally(results, output_path)

    print_final_results(success_count, len(results), output_path)


if __name__ == "__main__":
    main()