"""
maurice RLAIF Preference Synthesizer (maurice/synth.py)

Reads existing SFT datasets, generates responses using a strong base model (vLLM or HF),
scores them via LLM-as-a-Judge, and outputs a chosen/rejected preference dataset for DPO/ORPO.
"""

import argparse
import json
import logging
import os

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(filepath: str) -> list[dict]:
    data: list[dict] = []
    if not os.path.exists(filepath):
        logger.warning(f"File {filepath} not found.")
        return data
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item) + "\n" for item in data)
    logger.info(f"Saved {len(data)} preference pairs to {filepath}")


def generate_mock_responses(prompt: str, variant: str) -> list[str]:
    # In a real scenario, this would query vLLM or OpenAI API.
    # We mock generation for demonstration and CI stability.
    if variant == "c":
        return [
            "def solve():\n    return 'optimal solution'",
            "def solve():\n    # bad approach\n    pass",
        ]
    elif variant == "r":
        return [
            "<think>Step 1: Analyzing...</think> The answer is 42.",
            "I think it's 42, but I'm not sure.",
        ]
    else:
        return [
            "Here is a comprehensive and polite response to your query.",
            "I don't know the answer to that.",
        ]


def judge_responses(prompt: str, responses: list[str]) -> tuple[str, str]:
    # Mock LLM-as-a-Judge: we assume the first generated response is always better in our mock logic.
    # A real implementation would ask a Judge LLM to output a score from 1 to 10 for each response.
    return responses[0], responses[1]


def run_synthesis(variant: str, input_file: str | None = None, output_file: str | None = None, sample_size: int = -1):
    if not input_file:
        input_file = f"data/processed/train_{variant}.jsonl"
    if not output_file:
        output_file = f"data/processed/prefs_{variant}.jsonl"

    logger.info(f"Starting RLAIF Synthesis for variant '{variant}'")
    logger.info(f"Reading from {input_file}")

    dataset = load_jsonl(input_file)
    if not dataset:
        logger.error("Dataset is empty. Cannot synthesize preferences.")
        return

    if sample_size > 0:
        dataset = dataset[:sample_size]

    prefs_dataset = []

    for item in tqdm(dataset, desc="Synthesizing preferences"):
        # Extract user prompt from ChatML format
        prompt = ""
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        if not prompt:
            continue

        # Generate candidates
        candidates = generate_mock_responses(prompt, variant)

        # Score candidates
        chosen, rejected = judge_responses(prompt, candidates)

        # Create Preference Pair
        prefs_dataset.append(
            {
                "prompt": prompt,
                "chosen": [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}],
                "rejected": [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}],
            }
        )

    save_jsonl(prefs_dataset, output_file)
    logger.info("RLAIF Synthesis completed successfully.")


def main(args_list: list[str] | None = None):
    parser = argparse.ArgumentParser(description="maurice RLAIF Synthesizer")
    parser.add_argument("--variant", choices=["c", "r", "g"], required=True, help="Model variant")
    parser.add_argument("--input", type=str, help="Input SFT dataset (JSONL)")
    parser.add_argument("--output", type=str, help="Output preference dataset (JSONL)")
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples to process")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    args = parser.parse_args(args_list)

    run_synthesis(args.variant, args.input, args.output, args.samples)


if __name__ == "__main__":
    main()
