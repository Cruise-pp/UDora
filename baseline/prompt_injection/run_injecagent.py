"""
Prompt Injection Baseline — InjecAgent Runner

Evaluates the fixed-string prompt injection baseline on the InjecAgent dataset.
Data loading and progress utilities mirror injecagent.py for consistency.

Usage Example:
    # Direct-harm task, training split
    python baseline/prompt_injection/run_injecagent.py \\
        --model_id meta-llama/Llama-3.1-8B-Instruct --task dh --mode train --resume

    # Data-stealing task, training split
    python baseline/prompt_injection/run_injecagent.py \\
        --model_id meta-llama/Llama-3.1-8B-Instruct --task ds --mode train --resume
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

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

# huggingface login
# from huggingface_hub import login
# login(token="<your_hf_token_here>")  # Replace with your actual token


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt Injection Baseline — InjecAgent",
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
        default="train",
        choices=["train", "val"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ds",
        choices=["ds", "dh"],
        help="Task type: ds (Data Stealing) or dh (Direct Harm)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
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
        default="results/baseline/injecagent/",
        help="Directory to save attack results",
    )
    return parser.parse_args()


def get_data_path(model_name: str, mode: str, task: str) -> str:
    return f"data/injecagent/{model_name}_{mode}_{task}.json"


def generate_output_path(
    model_id: str, task: str, mode: str, output_dir: str
) -> str:
    sanitized = sanitize_filename(model_id)
    filename = f"{sanitized}_injecagent_{task}_{mode}.pkl"
    path = os.path.join(output_dir, mode)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)


def prepare_data(
    data_path: str, mode: str
) -> Tuple[List[str], List[str]]:
    """Load InjecAgent data and insert the {optim_str} placeholder."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "r") as f:
        data = json.load(f)

    if mode == "train":
        n = min(50, len(data))
        goals = [
            data[i]["model_input"].replace(
                data[i]["Attacker Instruction"],
                data[i]["Attacker Instruction"] + " {optim_str}",
            )
            for i in range(n)
        ]
        # prepend the {optim_str} for better performance
        # goals = [
        #     data[i]["model_input"].replace(
        #         data[i]["Attacker Instruction"],
        #         "{optim_str} " + data[i]["Attacker Instruction"],
        #     )
        #     for i in range(n)
        # ]
        targets = [data[i]["Attacker Tools"][0] for i in range(n)]
    else:
        last = len(data) - 1
        goals = [
            data[last]["model_input"].replace(
                data[last]["Attacker Instruction"],
                data[last]["Attacker Instruction"] + " {optim_str}",
            )
        ]
        targets = [data[last]["Attacker Tools"][0]]

    print(f"Loaded {len(goals)} examples for {mode} mode")
    return goals, targets


def main() -> None:
    args = parse_arguments()

    model_name = get_model_name(args.model_id)
    data_path = get_data_path(model_name, args.mode, args.task)
    output_path = generate_output_path(
        args.model_id, args.task, args.mode, args.output_dir
    )

    print(f"Model: {model_name} | Task: {args.task} | Mode: {args.mode}")
    print(f"Data:   {data_path}")
    print(f"Output: {output_path}")

    model, tokenizer = load_model_and_tokenizer(args.model_id)
    goals, targets = prepare_data(data_path, args.mode)

    attacker = PromptInjection(
        model, tokenizer, dataset="injecagent", max_new_tokens=args.max_new_tokens
    )

    results = []
    success_count = 0
    start_index = 0

    if args.resume:
        results, start_index, success_count = load_existing_results(output_path)

    for i in tqdm(range(start_index, len(goals)), desc="InjecAgent Prompt Injection"):
        result = attacker.run_single(goals[i], [targets[i]])
        success = get_success_status(result)
        success_count += int(success)
        print_attack_progress(
            i, len(goals), success, success_count, start_index,
            extra_info=f"Target: {targets[i]}",
        )
        results.append(result)
        save_results_incrementally(results, output_path)

    print_final_results(success_count, len(results), output_path)


if __name__ == "__main__":
    main()