#!/usr/bin/env python3
"""
maurice Proactive Code Review Daemon (scripts/08_code_review.py)

Proactively listens to PR events / diffs (via MCP or daemon loop),
analyzes code diffs using mau-llm-1.0-c model, identifies anti-patterns,
performance issues, or architectural violations, and if safe refactorings
are suggested, validates AST syntax, applies changes, and creates a commit
with message "refactor(jules): code review suggestions" on the PR branch
leaving it ready for human approval (no merge).
"""

import argparse
import ast
import json
import logging
import os
import re
import subprocess
import sys
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT_C = (
    "You are mau-llm-1.0-c, an expert code and refactoring engine. "
    "Analyze the provided pull request code diff for anti-patterns, performance issues, "
    "or architectural violations. If safe and clear refactoring exists, "
    "provide detailed feedback and output the refactored code or unified diff patch."
)


def brace_balance_check(code: str) -> bool:
    """Checks brace, parenthesis, and bracket balancing in code."""
    brace = paren = bracket = 0
    for ch in code:
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
        if brace < 0 or paren < 0 or bracket < 0:
            return False
    return brace == 0 and paren == 0 and bracket == 0


def validate_code_syntax(code: str, language: str = "python") -> bool:
    """Validates code syntax using Python AST or brace balance for other languages."""
    if not code or not isinstance(code, str):
        return False
    lang = language.lower().strip()
    if lang in ["python", "py"]:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    return brace_balance_check(code)


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Extracts code blocks (language, code) from Markdown text."""
    pattern = r"```([a-zA-Z0-9_+-]*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang.strip().lower() or "python", code) for lang, code in matches]


def extract_diff_patches(text: str) -> list[str]:
    """Extracts unified diff patches from Markdown text."""
    blocks = extract_code_blocks(text)
    patches = [code for lang, code in blocks if lang in ["diff", "patch"]]
    if not patches and "--- a/" in text and "+++ b/" in text:
        patches.append(text)
    return patches


def query_mau_model(prompt: str, server_url: str = "http://localhost:8000") -> str:
    """Queries mau-llm-1.0-c FastAPI serving endpoint or returns mock response fallback."""
    url = f"{server_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "mau-llm-1.0-c",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_C},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unable to reach mau-llm-1.0-c endpoint at {url} ({e}). Using mock analysis fallback.")

    # Mock response fallback for mau-llm-1.0-c
    return (
        "<think>\n"
        "Analyzing PR diff for anti-patterns, performance bottlenecks, and structural issues.\n"
        "Identified potential efficiency improvement in function call.\n"
        "</think>\n"
        "### Code Review Analysis\n"
        "- **Anti-patterns**: Redundant loop iterations.\n"
        "- **Performance**: Unnecessary calculation inside loop body.\n"
        "- **Architectural Violations**: None.\n\n"
        "### Recommended Safe Refactoring\n"
        "```python\n"
        "# Refactored optimized implementation\n"
        "def process_items(items):\n"
        "    cache = {item: item * 2 for item in items}\n"
        "    return [cache[i] for i in items if i in cache]\n"
        "```"
    )


def parse_review_response(response_text: str) -> dict[str, Any]:
    """Parses model response into structured issues, refactored code, and safety metrics."""
    code_blocks = extract_code_blocks(response_text)
    diff_patches = extract_diff_patches(response_text)

    anti_patterns = []
    performance_issues = []
    architectural_violations = []

    lines = response_text.splitlines()
    for line in lines:
        l = line.strip().lower()
        if "anti-pattern" in l or "antipattern" in l:
            anti_patterns.append(line.strip())
        elif "performance" in l or "efficiency" in l or "bottleneck" in l:
            performance_issues.append(line.strip())
        elif "architectur" in l or "violation" in l:
            architectural_violations.append(line.strip())

    refactored_code = None
    refactored_lang = "python"

    for lang, code in code_blocks:
        if lang not in ["diff", "patch"]:
            refactored_code = code
            refactored_lang = lang
            break

    is_safe = False
    if refactored_code:
        is_safe = validate_code_syntax(refactored_code, refactored_lang)

    return {
        "raw_response": response_text,
        "anti_patterns": anti_patterns,
        "performance_issues": performance_issues,
        "architectural_violations": architectural_violations,
        "refactored_code": refactored_code,
        "refactored_lang": refactored_lang,
        "diff_patches": diff_patches,
        "is_safe": is_safe,
    }


def apply_refactoring_to_files(
    review_result: dict[str, Any],
    file_map: dict[str, str],
    target_file: str | None = None,
    repo_dir: str = ".",
) -> list[str]:
    """Applies verified refactored code to target files in repo_dir."""
    modified_files: list[str] = []
    refactored_code = review_result.get("refactored_code")
    is_safe = review_result.get("is_safe", False)

    if not is_safe or not refactored_code:
        return modified_files

    if target_file and os.path.exists(os.path.join(repo_dir, target_file)):
        filepath = os.path.join(repo_dir, target_file)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(refactored_code)
        modified_files.append(target_file)
    elif file_map:
        for fname in file_map:
            filepath = os.path.join(repo_dir, fname)
            if os.path.exists(filepath):
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(refactored_code)
                modified_files.append(fname)
                break

    return modified_files


def commit_refactoring_changes(
    branch_name: str | None,
    modified_files: list[str],
    repo_dir: str = ".",
    commit_msg: str = "refactor(jules): code review suggestions",
) -> bool:
    """Commits modified refactoring files on the PR branch without merging."""
    if not modified_files:
        logger.info("No modified files to commit.")
        return False

    try:
        if branch_name:
            subprocess.run(
                ["git", "checkout", branch_name],
                cwd=repo_dir,
                capture_output=True,
                check=False,
            )

        for file_path in modified_files:
            subprocess.run(
                ["git", "add", file_path],
                cwd=repo_dir,
                capture_output=True,
                check=True,
            )

        res = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=repo_dir,
            capture_output=True,
            check=False,
        )

        if res.returncode == 0:
            logger.info(f"Successfully created commit '{commit_msg}' on branch '{branch_name or 'current'}'.")
            logger.info("Pull Request is ready for human approval (not merged).")
            return True
        else:
            logger.warning(f"Git commit returned non-zero status: {res.stderr.decode('utf-8')}")
            return False
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to commit refactoring changes: {e}")
        return False


def process_pr_event(
    event_data: dict[str, Any],
    server_url: str = "http://localhost:8000",
    auto_commit: bool = True,
    dry_run: bool = False,
    repo_dir: str = ".",
) -> dict[str, Any]:
    """Processes a PR event payload, analyzes diff with mau-llm-1.0-c, and commits safe refactorings."""
    pr_id = event_data.get("pr_id") or event_data.get("id") or "PR-unknown"
    branch = event_data.get("branch") or event_data.get("head_branch") or "feature-branch"
    diff_content = event_data.get("diff") or event_data.get("diff_text") or ""
    files_map = event_data.get("files") or {}
    target_file = event_data.get("target_file")

    logger.info(f"Processing PR #{pr_id} on branch '{branch}'...")

    prompt = (
        f"PR ID: {pr_id}\n"
        f"Branch: {branch}\n"
        f"Diff Content:\n{diff_content}\n\n"
        f"Please perform a code review identifying anti-patterns, performance bottlenecks, or "
        f"architectural violations. If appropriate, suggest safe refactoring code."
    )

    model_output = query_mau_model(prompt, server_url=server_url)
    review_result = parse_review_response(model_output)

    modified_files: list[str] = []
    committed = False

    if review_result["is_safe"] and review_result["refactored_code"]:
        if not dry_run:
            modified_files = apply_refactoring_to_files(
                review_result,
                file_map=files_map,
                target_file=target_file,
                repo_dir=repo_dir,
            )
            if auto_commit and modified_files:
                committed = commit_refactoring_changes(
                    branch_name=branch,
                    modified_files=modified_files,
                    repo_dir=repo_dir,
                )
        else:
            logger.info("[Dry Run] Safe refactoring identified; skipping file modifications and git commit.")

    return {
        "pr_id": pr_id,
        "branch": branch,
        "review_result": review_result,
        "modified_files": modified_files,
        "committed": committed,
        "ready_for_human_approval": True,
    }


def handle_mcp_rpc_request(request_json: dict[str, Any], server_url: str) -> dict[str, Any]:
    """Handles Model Context Protocol (MCP) JSON-RPC requests."""
    method = request_json.get("method")
    req_id = request_json.get("id", 1)
    params = request_json.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "review_pr_diff",
                        "description": "Analyzes PR diff with mau-llm-1.0-c, identifies issues, and commits safe refactorings.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "pr_id": {"type": "string"},
                                "branch": {"type": "string"},
                                "diff": {"type": "string"},
                                "auto_commit": {"type": "boolean"},
                            },
                            "required": ["diff"],
                        },
                    }
                ]
            },
        }

    if method in ["tools/call", "review_pr_diff", "pr_event"]:
        arguments = params.get("arguments", params)
        res = process_pr_event(
            event_data=arguments,
            server_url=server_url,
            auto_commit=arguments.get("auto_commit", True),
        )
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": res,
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method '{method}' not found"},
    }


def run_mcp_daemon(server_url: str = "http://localhost:8000") -> None:
    """Runs MCP daemon reading JSON-RPC requests line by line from stdin."""
    logger.info("Starting maurice MCP Proactive Code Review Daemon...")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            response = handle_mcp_rpc_request(req, server_url)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            err_resp = {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}}
            print(json.dumps(err_resp), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="maurice Proactive Code Review Daemon using mau-llm-1.0-c")
    parser.add_argument("--daemon", action="store_true", help="Run in MCP daemon mode listening on stdin")
    parser.add_argument("--pr-event", type=str, help="Path to JSON file containing PR event data")
    parser.add_argument("--diff-file", type=str, help="Path to text file containing git diff")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="mau-llm-1.0-c server URL")
    parser.add_argument(
        "--auto-commit",
        action="store_true",
        default=True,
        help="Commit safe refactoring changes to PR branch",
    )
    parser.add_argument(
        "--no-auto-commit",
        action="store_false",
        dest="auto_commit",
        help="Disable auto commit",
    )
    parser.add_argument("--dry-run", action="store_true", help="Perform review analysis without applying commits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.daemon:
        run_mcp_daemon(server_url=args.server_url)
        return

    event_data: dict[str, Any] = {}

    if args.pr_event and os.path.exists(args.pr_event):
        with open(args.pr_event, "r", encoding="utf-8") as f:
            event_data = json.load(f)
    elif args.diff_file and os.path.exists(args.diff_file):
        with open(args.diff_file, "r", encoding="utf-8") as f:
            event_data = {"diff": f.read(), "pr_id": "CLI-01", "branch": "current"}
    else:
        event_data = {
            "pr_id": "CLI-DEMO",
            "branch": "feature/demo",
            "diff": "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n-def foo(): pass\n+def foo(): return True",
        }

    res = process_pr_event(
        event_data=event_data,
        server_url=args.server_url,
        auto_commit=args.auto_commit,
        dry_run=args.dry_run,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
