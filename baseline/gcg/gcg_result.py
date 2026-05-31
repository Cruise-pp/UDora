"""
GCG Result Adapter

nanogcg.GCGResult only exposes best_loss, best_string, losses, and strings.
attack_utils.get_success_status() requires best_success and last_success.

This module bridges the gap by running a post-optimization generation for the
best and last adversarial suffixes and checking dataset-specific success
conditions — the same pattern used by baseline/prompt_injection/prompt_injection.py.
"""

import copy
from dataclasses import dataclass
from typing import List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from udora.datasets import check_success_condition


@dataclass
class GCGAdaptedResult:
    """GCG attack result compatible with attack_utils.get_success_status().

    Wraps the raw nanogcg.GCGResult fields and adds best_success / last_success
    so that the shared evaluation utilities work without modification.
    """

    # Preserved from nanogcg.GCGResult
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]

    # Last adversarial string (strings[-1] from the optimization trajectory)
    last_string: str

    # Model generations for the best and last adversarial suffixes
    best_generation: str
    last_generation: str

    # Success flags required by attack_utils.get_success_status()
    best_success: bool
    last_success: bool


def _substitute_optim_str(
    messages: Union[str, list], optim_str: str
) -> Union[str, list]:
    """Replace the {optim_str} placeholder with the actual adversarial string.

    Handles both plain-string prompts (InjecAgent) and chat-message lists
    (WebShop). Returns a copy — the original is never mutated.
    """
    if isinstance(messages, str):
        return messages.replace("{optim_str}", optim_str)

    messages = copy.deepcopy(messages)
    for msg in reversed(messages):
        if "{optim_str}" in msg.get("content", ""):
            msg["content"] = msg["content"].replace("{optim_str}", optim_str)
            break
    return messages


def _generate(
    messages: Union[str, list],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
) -> str:
    """Run greedy decoding on messages and return only the newly generated text."""
    if isinstance(messages, list):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        input_ids = tokenizer(
            messages, return_tensors="pt"
        ).input_ids.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids).to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = output_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_gcg_result(
    gcg_result,
    messages: Union[str, list],
    target: str,
    dataset: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
) -> GCGAdaptedResult:
    """Evaluate a nanogcg.GCGResult and return a GCGAdaptedResult.

    Generates model output for the best and last adversarial suffixes found
    during optimization, then checks dataset-specific success conditions.

    Args:
        gcg_result:      nanogcg.GCGResult returned by nanogcg.run()
        messages:        Original prompt with {optim_str} placeholder (str or list of dicts)
        target:          Attack target string
        dataset:         Dataset name ("injecagent" or "webshop")
        model:           The language model used for generation
        tokenizer:       Matching tokenizer
        max_new_tokens:  Maximum tokens to generate when checking success

    Returns:
        GCGAdaptedResult with best_success and last_success populated
    """
    best_string = gcg_result.best_string
    last_string = gcg_result.strings[-1] if gcg_result.strings else best_string

    # Generate and evaluate the best adversarial suffix
    best_messages = _substitute_optim_str(messages, best_string)
    best_generation = _generate(best_messages, model, tokenizer, max_new_tokens)
    best_success = check_success_condition(best_generation, [target], dataset)

    # Generate and evaluate the last adversarial suffix (skip if identical to best)
    if last_string == best_string:
        last_generation = best_generation
        last_success = best_success
    else:
        last_messages = _substitute_optim_str(messages, last_string)
        last_generation = _generate(last_messages, model, tokenizer, max_new_tokens)
        last_success = check_success_condition(last_generation, [target], dataset)

    return GCGAdaptedResult(
        best_loss=gcg_result.best_loss,
        best_string=best_string,
        losses=gcg_result.losses,
        strings=gcg_result.strings,
        last_string=last_string,
        best_generation=best_generation,
        last_generation=last_generation,
        best_success=best_success,
        last_success=last_success,
    )
