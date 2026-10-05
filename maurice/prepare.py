"""
maurice Dataset Preparation Pipeline (maurice/prepare.py)

Filters and formats datasets into ChatML JSONL format with calibrated <think> tags.
- Variant 'c' (Code & Refactor): Validates syntax (Python AST, C/Rust/Go patterns) and structures unified diff patches.
- Variant 'r' (Pure Reasoning): Preserves multi-step logic with explicit <think>...</think> blocks.
- Variant 'g' (General Purpose): Injects minimal <think>\n</think> tags for simple instructions to suppress overthinking.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import tempfile
from typing import Any

SYSTEM_PROMPTS = {
    "c": (
        "You are mau-llm-1.0-c, an expert code and refactoring engine. "
        "Provide clean, syntactically verified code, unified diff patches, "
        "and structural refactoring instructions."
    ),
    "r": (
        "You are mau-llm-1.0-r, a pure reasoning engine. "
        "Think carefully before answering by placing your step-by-step "
        "reasoning process inside <think>...</think> tags."
    ),
    "g": (
        "You are mau-llm-1.0-g, a versatile general-purpose assistant. "
        "Provide concise and accurate responses, using minimal thinking when appropriate."
    ),
}


def _brace_balance_check(code: str) -> bool:
    """Checks brace, parenthesis, and bracket balancing in code."""
    brace = paren = bracket = 0
    for ch in code:
        match ch:
            case "{":
                brace += 1
            case "}":
                brace -= 1
            case "(":
                paren += 1
            case ")":
                paren -= 1
            case "[":
                bracket += 1
            case "]":
                bracket -= 1
        if brace < 0 or paren < 0 or bracket < 0:
            return False
    return brace == 0 and paren == 0 and bracket == 0


def validate_js_ts_syntax(code: str, language: str = "javascript") -> bool:
    """Validates JavaScript/TypeScript syntax using node --check or falls back to brace balancing."""
    is_ts = language.lower() in ["typescript", "ts"]
    ext = ".ts" if is_ts else ".js"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        cmd = ["node", "--experimental-strip-types", tmp_path] if is_ts else ["node", "--check", tmp_path]
        result = subprocess.run(cmd, capture_output=True, timeout=5, check=False)
        if result.returncode == 0:
            return True

        if is_ts:
            res_check = subprocess.run(
                ["node", "--check", tmp_path],
                capture_output=True,
                timeout=5,
                check=False,
            )
            if res_check.returncode == 0:
                return True
            return _brace_balance_check(code)

        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return _brace_balance_check(code)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def validate_code_syntax(code: str, language: str = "python") -> bool:
    """Validates code syntax using ast for Python, node for JS/TS, and basic pattern checks for other C-like languages."""
    if not code or not isinstance(code, str):
        return False

    language = language.lower()
    if language in ["python", "py"]:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    elif language in ["javascript", "js", "typescript", "ts"]:
        return validate_js_ts_syntax(code, language)
    elif language in [
        "c",
        "cpp",
        "c++",
        "rust",
        "go",
        "java",
        "kotlin",
        "swift",
        "scala",
    ]:
        return _brace_balance_check(code)
    return True


def validate_think_tags(response: str) -> bool:
    """Validates explicit presence and structural ordering of <think>...</think> tags in response."""
    if not response or not isinstance(response, str):
        return False
    think_start = response.find("<think>")
    think_end = response.find("</think>")
    if think_start == -1 or think_end == -1:
        return False
    return think_start < think_end


def validate_code_in_assistant_content(content: str) -> bool:
    """Validates syntax of code blocks within assistant response content."""
    if not content or not isinstance(content, str):
        return False

    code_blocks = re.findall(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", content, re.DOTALL)
    if not code_blocks:
        return True

    for lang, code in code_blocks:
        lang = lang.strip().lower()
        if lang in [
            "python",
            "py",
            "javascript",
            "js",
            "typescript",
            "ts",
            "c",
            "cpp",
            "c++",
            "rust",
            "go",
            "java",
            "kotlin",
            "swift",
            "scala",
        ] and not validate_code_syntax(code, lang):
            return False
    return True


def format_chatml_example(system_prompt: str, user_prompt: str, assistant_response: str) -> dict[str, Any]:
    """Formats prompt components into standard ChatML schema."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.strip()},
            {"role": "assistant", "content": assistant_response.strip()},
        ]
    }


def generate_synthetic_samples(variant: str, count: int = 10) -> list[dict[str, Any]]:
    """Generates valid synthetic samples for testing and dry runs."""
    samples = []
    sys_prompt = SYSTEM_PROMPTS[variant]

    if variant == "c":
        for i in range(count):
            user_msg = f"Refactor Python function #{i + 1} to calculate factorial efficiently."
            code_diff = (
                "```diff\n"
                "--- a/math_utils.py\n"
                "+++ b/math_utils.py\n"
                "@@ -1,4 +1,4 @@\n"
                "-def factorial(n):\n"
                "-    return 1 if n <= 1 else n * factorial(n - 1)\n"
                "+import math\n"
                "+def factorial(n):\n"
                "+    return math.factorial(n)\n"
                "```"
            )
            resp = (
                "<think>\nAnalyzing factorial recursive call vs math.factorial binding in C-extension.\n"
                "Replacing recursion with standard math library for efficiency.\n</think>\n"
                f"Here is the refactored unified diff:\n\n{code_diff}"
            )
            samples.append(format_chatml_example(sys_prompt, user_msg, resp))

    elif variant == "r":
        for i in range(count):
            a, b = (i + 1) * 7, (i + 1) * 13
            user_msg = f"Solve for x: {a} + x = {b} * 2."
            rhs = b * 2
            ans = rhs - a
            resp = (
                f"<think>\n"
                f"Step 1: Calculate RHS: {b} * 2 = {rhs}.\n"
                f"Step 2: Set up equation: {a} + x = {rhs}.\n"
                f"Step 3: Subtract {a} from both sides: x = {rhs} - {a} = {ans}.\n"
                f"</think>\n"
                f"The value of x is {ans}."
            )
            samples.append(format_chatml_example(sys_prompt, user_msg, resp))

    elif variant == "g":
        for i in range(count):
            user_msg = f"What is the capital of country #{i + 1}?" if i > 0 else "What is the capital of France?"
            city = "Paris" if i == 0 else f"CapitalCity_{i + 1}"
            resp = f"<think>\n</think>\nThe capital is {city}."
            samples.append(format_chatml_example(sys_prompt, user_msg, resp))

    return samples


def process_hf_dataset(variant: str, sample_size: int = 50) -> list[dict[str, Any]]:
    """Loads and processes Hugging Face datasets for a specific variant."""
    records = []
    try:
        from datasets import load_dataset

        print(f"Attempting to load HF dataset for variant '{variant}'...")
        if variant == "c":
            ds = load_dataset(
                "bigcode/the-stack-smol-xs",
                data_dir="data/python",
                split=f"train[:{sample_size}]",
            )
            sys_prompt = SYSTEM_PROMPTS["c"]
            for row in ds:
                code = row.get("content", "")
                if validate_code_syntax(code, "python"):
                    records.append(
                        format_chatml_example(
                            sys_prompt,
                            "Validate and format the following code module:",
                            f"<think>\nParsing AST for Python code...\nValid syntax confirmed.\n</think>\n```python\n{code}\n```",
                        )
                    )
        elif variant == "r":
            ds = load_dataset(
                "HuggingFaceH4/Bespoke-Stratos-17k",
                split=f"train[:{sample_size}]",
            )
            sys_prompt = SYSTEM_PROMPTS["r"]
            for row in ds:
                conversations = row.get("conversations", [])
                user_val, assistant_val = "", ""
                if conversations and isinstance(conversations, list):
                    for msg in conversations:
                        if isinstance(msg, dict):
                            role = msg.get("role") or msg.get("from")
                            if role in ["user", "human"] and not user_val:
                                user_val = msg.get("value") or msg.get("content") or ""
                            elif role in ["assistant", "gpt"] and not assistant_val:
                                assistant_val = msg.get("value") or msg.get("content") or ""
                if user_val and assistant_val:
                    records.append(format_chatml_example(sys_prompt, user_val, assistant_val))
        elif variant == "g":
            ds = load_dataset("teknium/OpenHermes-2.5", split=f"train[:{sample_size}]")
            sys_prompt = SYSTEM_PROMPTS["g"]
            for row in ds:
                instruction = row.get("instruction", "")
                input_val = row.get("input", "")
                user_msg = f"{instruction}\n{input_val}".strip() if input_val else instruction.strip()
                output = row.get("output", "").strip()
                if user_msg and output:
                    if not output.startswith("<think>"):
                        output = f"<think>\n</think>\n{output}"
                    records.append(format_chatml_example(sys_prompt, user_msg, output))
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Failed to load HF dataset ({e}). Falling back to synthetic sample generation.")
        records = generate_synthetic_samples(variant, count=sample_size)
    return records


def process_variant(variant: str, output_path: str, sample_size: int = 50, synthetic: bool = True) -> int:
    """Processes dataset for a specific variant and writes to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records = []

    if synthetic:
        print(f"Generating synthetic records for variant '{variant}'...")
        records = generate_synthetic_samples(variant, count=sample_size)
    else:
        records = process_hf_dataset(variant, sample_size=sample_size)

    valid_records = []
    for record in records:
        msgs = record.get("messages", [])
        if (
            len(msgs) == 3
            and msgs[0]["role"] == "system"
            and msgs[1]["role"] == "user"
            and msgs[2]["role"] == "assistant"
        ):
            assistant_content = msgs[2].get("content", "")
            if variant == "r" and not validate_think_tags(assistant_content):
                continue
            if variant == "c" and not validate_code_in_assistant_content(assistant_content):
                continue
            valid_records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in valid_records)

    print(f"Successfully wrote {len(valid_records)} records to {output_path}")
    return len(valid_records)


def main(args_list: list[str] | None = None):
    parser = argparse.ArgumentParser(description="maurice Dataset Preparation Pipeline")
    parser.add_argument(
        "--variant",
        choices=["c", "r", "g", "all"],
        default="all",
        help="Target model variant",
    )
    parser.add_argument("--sample-size", type=int, default=50, help="Number of samples to process")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Use synthetic sample generator",
    )
    parser.add_argument(
        "--real-hf",
        action="store_false",
        dest="synthetic",
        help="Use real HuggingFace datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run with quick synthetic sample generation",
    )

    args = parser.parse_args(args_list)
    if args.dry_run:
        args.sample_size = min(args.sample_size, 5)
        args.synthetic = True
    variants = ["c", "r", "g"] if args.variant == "all" else [args.variant]

    total_prepared = 0
    for v in variants:
        out_file = f"data/processed/train_{v}.jsonl"
        count = process_variant(v, out_file, sample_size=args.sample_size, synthetic=args.synthetic)
        total_prepared += count

    print(f"Pipeline complete. Total records prepared across variants: {total_prepared}")


if __name__ == "__main__":
    main()
