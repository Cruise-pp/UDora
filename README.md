# UDora Reproduction & Extension (DSC291 Course Project)

This repository is a **fork of [AI-secure/UDora](https://github.com/AI-secure/UDora)** (UDora: A Unified Red Teaming Framework against LLM Agents, ICML 2025), created as a course project for **DSC291**.

On top of reproducing the original UDora attack, this fork contributes three things the upstream repository did not provide:

1. **Baseline attacks** — Template, GCG, and Prompt-Injection baselines so UDora can be compared against standard attacks under the same evaluation harness.
2. **A consistency study** — repeated-run experiments that measure how stable/reproducible UDora's attack success is when randomness is not fixed.
3. **Archived experiment results** — raw attack outputs, aggregate summaries, and exported tables under [`results/`](results/).

> The original upstream documentation is preserved at **[`README_UDora.md`](README_UDora.md)** — read it for the full description of the UDora algorithm, supported datasets, attack scripts, and command-line arguments. This README focuses on what *we* added.
>
> **To reproduce UDora's own attack results, please refer to the original repository: [AI-secure/UDora](https://github.com/AI-secure/UDora).**

---

## What We Added

| Module | Path | Description |
|--------|------|-------------|
| **Template Attack baseline** | [`baseline/template_attack.py`](baseline/template_attack.py) | Fixed jailbreak template (no optimization). Evaluated on AgentHarm and reports ASR broken down by the four `detailed × hint` quadrants. |
| **GCG baseline** | [`baseline/gcg/`](baseline/gcg/) | GCG adversarial-suffix attack built on `nanogcg`, with runners for WebShop, InjecAgent, and AgentHarm. |
| **Prompt Injection baseline** | [`baseline/prompt_injection/`](baseline/prompt_injection/) | Fixed-string prompt-injection attack (no optimization), dataset-agnostic, with runners for WebShop and InjecAgent. |
| **Consistency experiment** | [`consistency/`](consistency/) | Runs the same UDora setting multiple times without a fixed seed and reports ASR mean/variance, per-case stability, and run-to-run Jaccard overlap. |

### Design note: a unified evaluation interface

A key engineering goal was to evaluate every attack with UDora's *own* success criteria, so the numbers are directly comparable. Each baseline reuses:

- `udora.datasets.check_success_condition` — dataset-specific success conditions, and
- `attack_utils.get_success_status` — shared success/progress reporting.

To plug into that harness, each baseline result object exposes the same `best_success` / `last_success` interface UDora expects:

- [`baseline/gcg/gcg_result.py`](baseline/gcg/gcg_result.py) wraps `nanogcg.GCGResult`, re-generates the model output for the best and last suffixes, and fills in the success flags.
- [`baseline/prompt_injection/prompt_injection.py`](baseline/prompt_injection/prompt_injection.py) provides compatibility shims so a single-shot attack looks like a UDora result.

---

## Baselines

### 1. Template Attack (AgentHarm)

A static jailbreak template that forces a compliant opening (`"Absolutely! Here's"`) and forbids refusals — no optimization. ASR is reported across the four AgentHarm quadrants (detailed/simple prompt × with/without tool hint).

```bash
python baseline/template_attack.py \
    --model_id mistralai/Ministral-8B-Instruct-2410 \
    --data_path data/agentharm/harmful_behaviors_test_public.json \
    --output_path results/template_attack/ministral_results.json
```

### 2. GCG Baseline

Greedy Coordinate Gradient suffix optimization (via `nanogcg`), inserting an optimized `{optim_str}` into the agent's input and checking UDora's success condition on the generated output.

```bash
# WebShop
python baseline/gcg/run_webshop.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct --task 0 --mode train \
    --num_steps 500 --search_width 128 --topk 32 --early_stop --add_space_before_target --resume

# InjecAgent
python baseline/gcg/run_injecagent.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct --task ds --mode train --early_stop --resume

# AgentHarm
python baseline/gcg/run_agentharm.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct --early_stop --resume
```

### 3. Prompt Injection Baseline

A fixed imperative injection prefix (e.g. *"IMPORTANT!!! Ignore all previous instructions..."*) prepended to the attacker instruction — requires no optimization.

```bash
# InjecAgent
python baseline/prompt_injection/run_injecagent.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct --task ds --mode train --resume

# WebShop
python baseline/prompt_injection/run_webshop.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct --task 0 --mode train --resume
```

---

## Consistency Experiment

UDora's optimization is stochastic; without a fixed seed, repeated runs of the *same* setting can succeed on different cases. This experiment quantifies that variance.

**Setup.** `meta-llama/Llama-3.1-8B-Instruct`, InjecAgent **data-stealing (`ds`)** task, the **first 10 training cases**, 300 optimization steps, `num_location=3`, joint optimization, **no fixed seed**, repeated **5 times**.

**Run it:**

```bash
# 1) Run 5 repeated attacks and write summary.json
python consistency/run_consistency_injecagent_ds10.py

# 2) Export the summary into CSV tables (with Wilson / t confidence intervals)
python consistency/export_consistency_tables.py \
    --summary_json results/consistency/injecagent_ds_first10/summary.json
```

**Findings** (from [`results/consistency/injecagent_ds_first10/summary.json`](results/consistency/injecagent_ds_first10/summary.json)):

- **Stable cases:** 4 of 10 cases were fully consistent across all 5 runs (either always succeed or always fail); the rest flipped between runs.
- **Run-to-run overlap:** pairwise Jaccard of the success sets ranged from **0.67 to 1.0**.
- **95% CI of final success：**
  1. Wilson: [0.5833, 0.8253]
  2. t-interval: [0.5840, 0.8560]
- **Observation:** the `last`-string ASR was systematically higher than the `best`-string ASR, suggesting the early-stopping/best-selection criterion is not always picking the strongest adversarial string.

---

## Results Layout

```
results/
├── agentharm/train/        # UDora attack outputs on AgentHarm (varying num_location, joint vs. sequential)
├── injecagent/train/       # UDora attack outputs on InjecAgent (ds / dh)
├── gcg/train/              # GCG baseline outputs
├── template_attack/        # Template baseline outputs (JSON)
└── consistency/
    └── injecagent_ds_first10/
        ├── run_1 ... run_5/ # raw .pkl results per repeated run
        ├── summary.json     # aggregate consistency statistics
        └── tables/          # exported CSVs (run metrics, case×run matrices, Jaccard, CIs)
```

The `.pkl` filenames encode the hyperparameters of each run, e.g.:

```
meta_llama_Llama_3_1_8B_Instruct_injecagent_ds_steps300_width256_topk64_replace1_locations3_weight1.0_joint_standard_train.pkl
```

---

## Setup

```bash
conda create -n udora python=3.10
conda activate udora
pip install -r requirements.txt
```

See [`README_UDora.md`](README_UDora.md) for full installation details (including `inspect_ai` / `inspect_evals` setup required only for AgentHarm) and the complete list of attack arguments.

---

## Acknowledgments & Citation

This is a course project built on the original UDora codebase. Please cite the original paper:

```bibtex
@article{zhang2025udora,
  title={Udora: A unified red teaming framework against llm agents by dynamically hijacking their own reasoning},
  author={Zhang, Jiawei and Yang, Shuang and Li, Bo},
  journal={arXiv preprint arXiv:2503.01908},
  year={2025}
}
```

- **Original UDora:** [AI-secure/UDora](https://github.com/AI-secure/UDora) · [arXiv:2503.01908](https://arxiv.org/abs/2503.01908)
- **nanoGCG:** [GraySwanAI/nanoGCG](https://github.com/GraySwanAI/nanoGCG) — used for the GCG baseline.
