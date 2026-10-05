"""
scripts/08_mcp_quality_gates.py
Automates Quality & Validation Gates via local MCP server and GitHub PR webhooks.

Capabilities:
1. Static validation of Containerfile: "Nunca usar musl (Alpine) se a imagem for baseada em glibc".
2. Quality Gates execution: ruff check --fix, ruff format, pytest tests/ -v -m "not slow".
3. Auto-commits formatting fixes if changes are made.
4. Returns failure logs and suggested diff comments if tests fail irremediably.
5. FastAPI / MCP JSON-RPC endpoints for handling PR webhooks and MCP tool calls.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_containerfile(filepath: str = "Containerfile") -> tuple[bool, str]:
    """
    Statically validates Containerfile to enforce glibc binary runtime compatibility rule:
    'Nunca usar musl (Alpine) se a imagem for baseada em glibc'.
    """
    if not os.path.exists(filepath):
        return True, f"Containerfile '{filepath}' not found. Skipping static container check."

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:  # noqa: BLE001
        return False, f"Failed to read Containerfile '{filepath}': {e}"

    violations = []
    has_musl_alpine = False

    for idx, line in enumerate(lines, 1):
        clean_line = line.strip()
        if clean_line.startswith("#") or not clean_line:
            continue

        upper_line = clean_line.upper()

        if upper_line.startswith("FROM"):
            image_spec = clean_line.split()[1].lower() if len(clean_line.split()) > 1 else ""
            if "alpine" in image_spec or "musl" in image_spec:
                has_musl_alpine = True
                violations.append(f"Line {idx}: Image '{image_spec}' uses musl/Alpine.")

        if "apk add" in clean_line.lower() or "apk update" in clean_line.lower():
            has_musl_alpine = True
            violations.append(f"Line {idx}: Package manager 'apk' detected (Alpine/musl).")

        if "musl" in clean_line.lower() and not upper_line.startswith("LABEL") and not upper_line.startswith("#"):
            has_musl_alpine = True
            violations.append(f"Line {idx}: Reference to 'musl' C-runtime detected.")

    if has_musl_alpine:
        err_msg = (
            "Containerfile rule violation: Found musl/Alpine usage in Containerfile.\n"
            "Rule: 'Nunca usar musl (Alpine) se a imagem for baseada em glibc'.\n"
            "Violations:\n" + "\n".join(f"- {v}" for v in violations) + "\n"
            "Remediation: Ensure all stages use glibc-compatible base images (e.g., debian:bookworm-slim, python:3.11-slim-bookworm)."
        )
        return False, err_msg

    return True, "Containerfile static check passed: fully compliant with glibc runtime rules."


def get_git_diff(repo_path: str = ".") -> str:
    """Returns the output of `git diff` for the repository."""
    try:
        res = subprocess.run(
            ["git", "diff"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        return res.stdout
    except Exception as e:  # noqa: BLE001
        return f"Error retrieving git diff: {e}"


def check_git_status(repo_path: str = ".") -> bool:
    """Checks whether git working tree has modified or untracked changes."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(res.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def commit_changes(repo_path: str = ".", commit_msg: str = "style: auto-fix quality gates (ruff)") -> tuple[bool, str]:
    """Stages all changes and commits them directly to the active branch."""
    try:
        add_res = subprocess.run(
            ["git", "add", "-A"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if add_res.returncode != 0:
            return False, f"git add failed: {add_res.stderr}"

        commit_res = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_res.returncode != 0:
            return False, f"git commit failed: {commit_res.stderr}"

        return True, commit_res.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return False, f"Exception during git commit: {e}"


def run_quality_gates(
    repo_path: str = ".",
    auto_commit: bool = True,
    containerfile_path: str = "Containerfile",
) -> dict[str, Any]:
    """
    Executes full quality and validation gate checks:
    1. Static Containerfile glibc vs musl check.
    2. ruff check --fix .
    3. ruff format .
    4. pytest tests/ -v -m "not slow"
    5. Git auto-commit if ruff formatted/fixed files.
    """
    logs: list[str] = []
    logs.append("==> Running Quality & Validation Gates...")

    # Step 1: Static Containerfile check
    cf_passed, cf_msg = validate_containerfile(containerfile_path)
    logs.append(f"1. Containerfile Static Check: {'PASSED' if cf_passed else 'FAILED'}")
    logs.append(f"   {cf_msg}")

    # Step 2: ruff check --fix .
    ruff_check_res = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--fix", "."],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    ruff_check_passed = ruff_check_res.returncode == 0
    logs.append(f"2. Ruff Check Fix: {'PASSED' if ruff_check_passed else 'FAILED'}")
    if ruff_check_res.stdout:
        logs.append(f"   stdout: {ruff_check_res.stdout.strip()}")
    if ruff_check_res.stderr:
        logs.append(f"   stderr: {ruff_check_res.stderr.strip()}")

    # Step 3: ruff format .
    ruff_fmt_res = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "."],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    ruff_fmt_passed = ruff_fmt_res.returncode == 0
    logs.append(f"3. Ruff Format: {'PASSED' if ruff_fmt_passed else 'FAILED'}")
    if ruff_fmt_res.stdout:
        logs.append(f"   stdout: {ruff_fmt_res.stdout.strip()}")
    if ruff_fmt_res.stderr:
        logs.append(f"   stderr: {ruff_fmt_res.stderr.strip()}")

    # Check for git changes (formatting / linter auto-fixes)
    has_changes = check_git_status(repo_path)
    auto_committed = False
    commit_msg_out = ""

    if has_changes and auto_commit:
        commit_success, commit_msg_out = commit_changes(
            repo_path, "style: auto-fix formatting and linting via MCP quality gates"
        )
        if commit_success:
            auto_committed = True
            logs.append(f"4. Auto-commit: Executed successfully ({commit_msg_out})")
        else:
            logs.append(f"4. Auto-commit: Failed ({commit_msg_out})")

    # Step 4: pytest tests/ -v -m "not slow"
    pytest_res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "-m", "not slow"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    pytest_passed = pytest_res.returncode == 0
    logs.append(f"5. Pytest Suite (not slow): {'PASSED' if pytest_passed else 'FAILED'}")

    overall_success = cf_passed and ruff_check_passed and ruff_fmt_passed and pytest_passed

    error_log = ""
    suggested_diff = ""

    if not overall_success:
        err_parts = []
        if not cf_passed:
            err_parts.append(f"Containerfile check failed:\n{cf_msg}")
        if not ruff_check_passed:
            err_parts.append(f"Ruff check failed:\n{ruff_check_res.stderr or ruff_check_res.stdout}")
        if not ruff_fmt_passed:
            err_parts.append(f"Ruff format failed:\n{ruff_fmt_res.stderr or ruff_fmt_res.stdout}")
        if not pytest_passed:
            err_parts.append(f"Pytest suite failed:\n{pytest_res.stdout}\n{pytest_res.stderr}")

        error_log = "\n---\n".join(err_parts)
        suggested_diff = get_git_diff(repo_path)

    result = {
        "success": overall_success,
        "containerfile_valid": cf_passed,
        "ruff_check_passed": ruff_check_passed,
        "ruff_format_passed": ruff_fmt_passed,
        "pytest_passed": pytest_passed,
        "auto_committed": auto_committed,
        "commit_info": commit_msg_out if auto_committed else None,
        "error_log": error_log,
        "suggested_diff": suggested_diff,
        "logs": "\n".join(logs),
    }

    return result


class PRWebhookPayload(BaseModel):
    action: str = "opened"
    pull_request: dict[str, Any] = Field(default_factory=dict)
    repository: dict[str, Any] = Field(default_factory=dict)


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: str | int | None = 1


app = FastAPI(title="maurice Quality & Validation Gates MCP Server")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "maurice-mcp-quality-gates"}


@app.post("/webhook/pull-request")
@app.post("/webhook/pr")
async def handle_pr_webhook(payload: PRWebhookPayload):
    """
    Intercepts PR creation/update events ('opened', 'synchronize', 'reopened').
    Runs Quality & Validation Gates locally.
    """
    logger.info(f"Received PR webhook event. Action: '{payload.action}'")
    valid_actions = {"opened", "synchronize", "reopened"}

    if payload.action not in valid_actions:
        return {"status": "ignored", "message": f"Action '{payload.action}' does not trigger quality gates."}

    pr_number = payload.pull_request.get("number")
    gate_results = run_quality_gates(auto_commit=True)

    if gate_results["success"]:
        return {
            "status": "success",
            "message": "Quality & validation gates passed successfully.",
            "pr_number": pr_number,
            "auto_committed": gate_results["auto_committed"],
            "logs": gate_results["logs"],
        }
    else:
        return JSONResponse(
            status_code=422,
            content={
                "status": "failure",
                "message": "Quality & validation gates failed.",
                "pr_number": pr_number,
                "comment_suggestion": {
                    "log": gate_results["error_log"],
                    "suggested_diff": gate_results["suggested_diff"],
                },
                "logs": gate_results["logs"],
            },
        )


@app.post("/mcp")
async def handle_mcp_jsonrpc(request: MCPRequest):
    """
    Standard MCP (Model Context Protocol) JSON-RPC handler supporting:
    - initialize
    - tools/list
    - tools/call
    - ping
    """
    method = request.method
    params = request.params

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request.id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "maurice-mcp-quality-gates", "version": "1.0.0"},
            },
        }

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": request.id, "result": {}}

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request.id,
            "result": {
                "tools": [
                    {
                        "name": "run_quality_gates",
                        "description": "Runs full quality & validation gates (Containerfile glibc check, ruff check/format, pytest).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "repo_path": {"type": "string", "default": "."},
                                "auto_commit": {"type": "boolean", "default": True},
                                "containerfile_path": {"type": "string", "default": "Containerfile"},
                            },
                        },
                    },
                    {
                        "name": "validate_containerfile",
                        "description": "Statically validates Containerfile to ensure glibc vs musl/Alpine compliance.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "filepath": {"type": "string", "default": "Containerfile"},
                            },
                        },
                    },
                    {
                        "name": "handle_pr_event",
                        "description": "Handles PR webhook event payloads (opened, synchronize, reopened).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                                "pull_request": {"type": "object"},
                            },
                            "required": ["action"],
                        },
                    },
                ]
            },
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "run_quality_gates":
            res = run_quality_gates(
                repo_path=arguments.get("repo_path", "."),
                auto_commit=arguments.get("auto_commit", True),
                containerfile_path=arguments.get("containerfile_path", "Containerfile"),
            )
            return {
                "jsonrpc": "2.0",
                "id": request.id,
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
            }

        elif tool_name == "validate_containerfile":
            filepath = arguments.get("filepath", "Containerfile")
            passed, msg = validate_containerfile(filepath)
            return {
                "jsonrpc": "2.0",
                "id": request.id,
                "result": {"content": [{"type": "text", "text": json.dumps({"valid": passed, "message": msg})}]},
            }

        elif tool_name == "handle_pr_event":
            action = arguments.get("action", "opened")
            if action not in {"opened", "synchronize", "reopened"}:
                res = {"status": "ignored", "message": f"Action '{action}' ignored."}
            else:
                res = run_quality_gates(auto_commit=True)

            return {
                "jsonrpc": "2.0",
                "id": request.id,
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: '{tool_name}'")

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported MCP method: '{method}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="maurice MCP Quality & Validation Gate Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--run-gates", action="store_true", help="Run quality gates directly from CLI")
    parser.add_argument("--no-commit", action="store_true", help="Disable auto-committing format fixes")
    parser.add_argument("--containerfile", type=str, default="Containerfile", help="Path to Containerfile")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_gates:
        res = run_quality_gates(
            auto_commit=not args.no_commit,
            containerfile_path=args.containerfile,
        )
        print(res["logs"])
        if not res["success"]:
            print("\nErrors / Diff Suggested:\n", res["error_log"])
            sys.exit(1)
        sys.exit(0)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        parse_args()
    main()
