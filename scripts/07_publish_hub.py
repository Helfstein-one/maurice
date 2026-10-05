#!/usr/bin/env python3
"""
maurice Automated Hugging Face Hub Publishing Pipeline (scripts/07_publish_hub.py)

Packages and publishes merged 16-bit FP16 models and quantized GGUF binaries to Hugging Face Hub,
dynamically generating standardized Model Cards (README.md) with benchmark performance metrics.
"""

import argparse
import json
import os
from typing import Any

VARIANT_DESCRIPTIONS = {
    "c": "Code refactoring and syntax-grounded diff generation.",
    "r": "Chain-of-thought mathematical reasoning (<think> calibration).",
    "g": "General-purpose instruction following with adaptive thinking.",
}

VARIANT_NAMES = {
    "c": "mau-llm-1.0-c (Code)",
    "r": "mau-llm-1.0-r (Reasoning)",
    "g": "mau-llm-1.0-g (General)",
}


def parse_benchmark_results(benchmark_file: str, variant: str) -> dict[str, Any] | None:
    """Parses benchmark_results.json and returns metrics for the specified variant if available."""
    if not benchmark_file or not os.path.exists(benchmark_file):
        return None

    try:
        with open(benchmark_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            target_variant = f"mau-llm-1.0-{variant}"
            for item in data:
                if isinstance(item, dict) and item.get("variant") == target_variant:
                    return item.get("metrics")
        elif isinstance(data, dict):
            if "variant_summaries" in data:
                target_variant = f"mau-llm-1.0-{variant}"
                for summary in data["variant_summaries"]:
                    if summary.get("variant") == target_variant:
                        return summary
            elif data.get("variant") == f"mau-llm-1.0-{variant}":
                return data.get("metrics")
    except Exception as e:  # noqa: BLE001
        print(f"[Warning] Failed to parse benchmark file {benchmark_file}: {e}")

    return None


def generate_model_card(variant: str, benchmark_file: str = "build/benchmark_results.json") -> str:
    """Generates a rich README.md Model Card with YAML metadata and benchmark metrics."""
    desc = VARIANT_DESCRIPTIONS.get(variant, "Specialized maurice language model variant.")
    variant_title = VARIANT_NAMES.get(variant, f"mau-llm-1.0-{variant}")

    # Parse dynamic metrics if available
    metrics = parse_benchmark_results(benchmark_file, variant)

    if metrics:
        tps = metrics.get("tokens_per_second", "N/A")
        ttft = metrics.get("time_to_first_token_ms", "N/A")
        rss = metrics.get("peak_rss_mb", "N/A")
        eval_scores = metrics.get("evaluation_scores", {})
        eval_str = ", ".join(f"{k}: {v}" for k, v in eval_scores.items()) if eval_scores else "N/A"

        benchmark_table = f"""| Metric | Value |
| :--- | :--- |
| **Throughput (tokens/sec)** | `{tps}` |
| **Time to First Token (TTFT)** | `{ttft} ms` |
| **Peak Memory RSS** | `{rss} MB` |
| **Domain Evaluation** | `{eval_str}` |"""
    else:
        benchmark_table = """*Benchmark evaluation results pending or unavailable at publication time.*"""

    model_card = f"""---
license: mit
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
tags:
- deepseek
- qwen
- code
- gguf
- mlx
- llama.cpp
---

# maurice 1.5B - {variant_title}

**maurice** (*Minimal Adaptation for Ultra-fast Reasoning and Inference in Code Engines*) is an end-to-end 1.5B parameter language model series fine-tuned from `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

## Variant Details

- **Variant**: `mau-llm-1.0-{variant}`
- **Specialization**: {desc}
- **Base Architecture**: DeepSeek-R1-Distill-Qwen-1.5B

## Prompt Format

This model uses the standard **ChatML** prompt structure with support for reasoning blocks (`<think>`):

```
<|im_start|>system
You are maurice, an expert AI assistant.
<|im_end|>
<|im_start|>user
{{prompt}}
<|im_end|>
<|im_start|>assistant
<think>
{{reasoning}}
</think>
{{response}}
<|im_end|>
```

## Hardware & Inference Performance Benchmark

{benchmark_table}

## Usage

### llama.cpp / GGUF

```bash
llama-cli -m mau-llm-1.0-{variant}-q4_k_m.gguf -p "<|im_start|>user\\nHello!\\n<|im_end|>\\n<|im_start|>assistant\\n"
```

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Helfstein-one/mau-llm-1.0-{variant}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```
"""
    return model_card.strip() + "\n"


def publish_variant(
    variant: str,
    repo_id: str,
    weights_dir: str,
    gguf_dir: str,
    token: str | None = None,
    private: bool = False,
    dry_run: bool = False,
    benchmark_file: str = "build/benchmark_results.json",
) -> None:
    """Publishes weights, GGUF binaries, and Model Card for a single variant."""
    weights_path = weights_dir.format(variant=variant) if "{variant}" in weights_dir else weights_dir
    gguf_path = os.path.join(gguf_dir, f"mau-llm-1.0-{variant}-q4_k_m.gguf")

    print("==================================================")
    print(f"Publishing maurice Model Variant: '{variant}'")
    print(f"Target HF Repository: {repo_id}")
    print(f"Merged Weights Path: {weights_path}")
    print(f"Quantized GGUF Path: {gguf_path}")
    print(f"Private Repository: {private}")
    print(f"Dry-Run Mode: {dry_run}")
    print("==================================================")

    # Generate Model Card
    readme_content = generate_model_card(variant, benchmark_file)

    if dry_run:
        print("[Dry-Run] Model Card generated successfully. Preview:")
        print("--------------------------------------------------")
        print(readme_content[:400] + "\n...")
        print("--------------------------------------------------")

        if os.path.exists(weights_path):
            print(f"[Dry-Run] Merged weights directory found at {weights_path}.")
        else:
            print(f"[Dry-Run / Notice] Merged weights directory not found at {weights_path} (will skip in dry-run).")

        if os.path.exists(gguf_path):
            print(f"[Dry-Run] Quantized GGUF binary found at {gguf_path}.")
        else:
            print(f"[Dry-Run / Notice] Quantized GGUF binary not found at {gguf_path} (will skip in dry-run).")

        print(f"[Dry-Run] Simulated successful publication for repository '{repo_id}'.")
        return

    # Real publication with huggingface_hub
    if not token:
        token = os.environ.get("HF_TOKEN")

    if not token:
        raise ValueError(
            "Hugging Face API token is required for publication. "
            "Provide via --token argument or set HF_TOKEN environment variable."
        )

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

        # Create repo if not exists
        print(f"Ensuring repository '{repo_id}' exists on Hugging Face Hub...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

        # Upload 16-bit safetensors/weights folder if it exists
        if os.path.exists(weights_path) and os.path.isdir(weights_path):
            print(f"Uploading merged 16-bit model files from {weights_path}...")
            api.upload_folder(
                folder_path=weights_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print("16-bit model files uploaded successfully.")

        # Upload quantized GGUF binary if it exists
        if os.path.exists(gguf_path):
            print(f"Uploading quantized GGUF binary from {gguf_path}...")
            api.upload_file(
                path_or_fileobj=gguf_path,
                path_in_repo=f"mau-llm-1.0-{variant}-q4_k_m.gguf",
                repo_id=repo_id,
                repo_type="model",
            )
            print("Quantized GGUF binary uploaded successfully.")

        # Upload Model Card README.md
        print("Uploading Model Card (README.md)...")
        readme_bytes = readme_content.encode("utf-8")
        api.upload_file(
            path_or_fileobj=readme_bytes,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Publication of variant '{variant}' to '{repo_id}' completed successfully!")

    except Exception as e:
        print(f"[Error] Failed to publish variant '{variant}' to Hugging Face Hub: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="maurice Hugging Face Hub Publisher")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Destination Hugging Face repository ID (e.g. Helfstein-one/mau-llm-1.0-c)",
    )
    parser.add_argument(
        "--variant",
        choices=["c", "r", "g", "all"],
        default="c",
        help="Model variant to publish (default: c)",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Path to merged weights directory (default: checkpoints/merged_{variant})",
    )
    parser.add_argument(
        "--gguf-dir",
        type=str,
        default="build",
        help="Path to quantized GGUF directory (default: build)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API auth token (default: reads HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update repository as private",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate artifacts and simulate upload without making network requests",
    )
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default="build/benchmark_results.json",
        help="Path to benchmark results JSON file",
    )

    args = parser.parse_args()

    # Required check for repo-id unless dry-run
    if not args.repo_id and not args.dry_run and args.variant != "all":
        # If repo_id not provided, set default pattern
        args.repo_id = f"Helfstein-one/mau-llm-1.0-{args.variant}"

    variants = ["c", "r", "g"] if args.variant == "all" else [args.variant]

    for v in variants:
        v_weights_dir = args.weights_dir or f"checkpoints/merged_{v}"
        if args.repo_id and "{variant}" in args.repo_id:
            v_repo_id = args.repo_id.format(variant=v)
        elif args.repo_id and args.variant == "all":
            # Append variant or use base
            v_repo_id = f"{args.repo_id}-{v}" if not args.repo_id.endswith(f"-{v}") else args.repo_id
        elif args.repo_id:
            v_repo_id = args.repo_id
        else:
            v_repo_id = f"Helfstein-one/mau-llm-1.0-{v}"

        publish_variant(
            variant=v,
            repo_id=v_repo_id,
            weights_dir=v_weights_dir,
            gguf_dir=args.gguf_dir,
            token=args.token,
            private=args.private,
            dry_run=args.dry_run,
            benchmark_file=args.benchmark_file,
        )


if __name__ == "__main__":
    main()
