"""
Common Utilities for Attack Scripts

This module contains shared utility functions used across different attack scripts
(webshop.py, injecagent.py, agentharm.py) to reduce code duplication.

Common functionality includes:
- Model loading and configuration
- File path operations
- Resume functionality
- Result saving and loading
"""

import os
import re
import pickle
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_name(model_id: str) -> str:
    """Map model ID to simplified model name for data path determination.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Simplified model name (llama, ministral, or sanitized version)
    """
    if 'llama' in model_id.lower():
        return 'llama'
    elif 'ministral' in model_id.lower():
        return 'ministral'
    else:
        return model_id.replace('/', '_')


def sanitize_filename(name: str) -> str:
    """Create filesystem-friendly filename from model ID.
    
    Args:
        name: Original filename or model ID
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    return re.sub(r'[^A-Za-z0-9]+', '_', name)


def load_model_and_tokenizer(model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load and configure model and tokenizer with consistent settings.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Tuple of (model, tokenizer) ready for attack
    """
    print(f"Loading model: {model_id}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Ensure tokenizer has required tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def create_output_directory(output_path: str) -> None:
    """Create output directory structure if it doesn't exist.
    
    Args:
        output_path: Full path to output file
    """
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)


def load_existing_results(output_path: str, verbose: bool = True) -> Tuple[List[Any], int, int]:
    """Load existing attack results for resume functionality.
    
    Args:
        output_path: Path to existing results file
        verbose: Whether to print status messages
        
    Returns:
        Tuple of (results_list, start_index, success_count)
    """
    results = []
    start_index = 0
    success_count = 0
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'rb') as f:
                results = pickle.load(f)
            start_index = len(results)
            
            # Recalculate success rate from existing results
            for res in results:
                success = get_success_status(res)
                success_count += int(success)
                
            if verbose:
                print(f"Resuming from index {start_index}. Current ASR: {success_count}/{start_index}")
        except Exception as e:
            if verbose:
                print(f"Failed to load existing results: {e}")
            results = []
            start_index = 0
            success_count = 0
    
    return results, start_index, success_count


def get_success_status(result) -> bool:
    """Extract success status from UDora result object.
    
    Args:
        result: UDoraResult object
        
    Returns:
        Boolean indicating if attack was successful
    """
    # Handle both single and list results
    if isinstance(result.best_success, list):
        best_success = bool(result.best_success[0] if result.best_success else False)
        last_success = bool(result.last_success[0] if result.last_success else False)
    else:
        best_success = bool(result.best_success or False)
        last_success = bool(result.last_success or False)
    
    return best_success or last_success


def save_results_incrementally(results: List[Any], output_path: str) -> None:
    """Save attack results incrementally to prevent data loss.
    
    Args:
        results: List of attack results
        output_path: Path to save results
    """
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)


def print_attack_progress(
    current_idx: int, 
    total_count: int, 
    success: bool, 
    success_count: int, 
    start_index: int = 0,
    extra_info: str = ""
) -> None:
    """Print standardized attack progress information.
    
    Args:
        current_idx: Current attack index (0-based)
        total_count: Total number of attacks
        success: Whether current attack succeeded
        success_count: Total number of successful attacks
        start_index: Starting index for resumed attacks
        extra_info: Additional information to display
    """
    progress_info = f"Goal {current_idx+1}/{total_count} - Success: {success} - ASR: {success_count}/{current_idx+1-start_index}"
    if extra_info:
        progress_info += f" - {extra_info}"
    print(progress_info)


def print_final_results(success_count: int, total_results: int, output_path: str) -> None:
    """Print final attack results summary.
    
    Args:
        success_count: Number of successful attacks
        total_results: Total number of attack results
        output_path: Path where results were saved
    """
    final_asr = success_count / total_results if total_results else 0
    print(f"Attack completed. Final ASR: {success_count}/{total_results} ({final_asr:.2%})")
    print(f"Results saved to: {output_path}")
