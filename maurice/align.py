"""
maurice Post-SFT Alignment Trainer (maurice/align.py)

Aligns the fine-tuned models using DPO (Direct Preference Optimization) or ORPO.
Reads from data/processed/prefs_{variant}.jsonl
"""

import argparse
import json
import logging
import os
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_variant_config(variant: str, config_dir: str = "configs") -> dict[str, Any]:
    config_file = os.path.join(config_dir, f"variant_{variant}.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file for variant '{variant}' not found at {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def run_alignment(
    variant: str,
    method: str = "orpo",
    dataset_file: str | None = None,
    output_dir: str | None = None,
    dry_run: bool = False,
):
    if not dataset_file:
        dataset_file = f"data/processed/prefs_{variant}.jsonl"

    if not output_dir:
        output_dir = f"checkpoints/aligned_{variant}_{method}"

    logger.info("==================================================")
    logger.info(f"Starting {method.upper()} Post-SFT Alignment for variant: {variant}")
    logger.info(f"Dataset: {dataset_file}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("==================================================")

    if dry_run:
        logger.info(f"[DRY-RUN] Would load config for variant {variant}")
        logger.info(f"[DRY-RUN] Would load dataset {dataset_file}")
        logger.info(f"[DRY-RUN] Would initialize {method.upper()}Trainer from TRL")
        logger.info(f"[DRY-RUN] Would train and save to {output_dir}")
        logger.info("[DRY-RUN] Alignment dry-run completed successfully.")
        return

    # In a real run, you would load the model, peft config, and initialize DPOTrainer or ORPOTrainer
    # Example logic:
    # from trl import DPOTrainer, ORPOTrainer
    # ...
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Alignment complete (mock execution).")


def main(args_list: list[str] | None = None):
    parser = argparse.ArgumentParser(description="maurice Post-SFT Alignment")
    parser.add_argument("--variant", choices=["c", "r", "g"], required=True, help="Model variant")
    parser.add_argument("--method", choices=["dpo", "orpo"], default="orpo", help="Alignment method")
    parser.add_argument("--dataset", type=str, help="Input preference dataset (JSONL)")
    parser.add_argument("--output-dir", type=str, help="Directory to save aligned checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Run without executing heavy computations")

    args = parser.parse_args(args_list)
    run_alignment(
        variant=args.variant,
        method=args.method,
        dataset_file=args.dataset,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
