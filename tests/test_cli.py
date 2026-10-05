import subprocess
import sys
from unittest.mock import patch

import pytest

from maurice import __version__, cli


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert f"maurice {__version__}" in captured.out


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    for sub in ["prepare", "train", "merge", "quantize", "eval", "serve", "ui", "info"]:
        assert sub in captured.out


def test_cli_no_args(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage: maurice" in captured.out


def test_cli_info(capsys):
    cli.main(["info"])
    captured = capsys.readouterr()
    assert f"maurice Package Version: {__version__}" in captured.out
    assert "Available Model Variants:" in captured.out


def test_cli_prepare_dry_run(capsys):
    cli.main(["prepare", "--variant", "c", "--sample-size", "2", "--dry-run"])
    captured = capsys.readouterr()
    assert "Generating synthetic records" in captured.out
    assert "Pipeline complete" in captured.out


def test_cli_train_dry_run(capsys):
    cli.main(["train", "--variant", "c", "--dry-run"])
    captured = capsys.readouterr()
    assert "Starting QLoRA Fine-Tuning" in captured.out
    assert "[Dry-Run Mode]" in captured.out


def test_cli_merge_dry_run(capsys):
    cli.main(["merge", "--variant", "c", "--dry-run"])
    captured = capsys.readouterr()
    assert "Starting Weight Merge" in captured.out
    assert "[Dry-Run / Fallback Mode]" in captured.out


def test_cli_eval_dry_run(capsys, tmp_path):
    out_json = str(tmp_path / "bench.json")
    cli.main(["eval", "--variant", "c", "--dry-run", "--output-json", out_json])
    captured = capsys.readouterr()
    assert "Running Hardware Benchmark & Evaluation" in captured.out
    assert "mau-llm-1.0-c" in captured.out


def test_cli_quantize(capsys):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["quantize", "--variant", "c", "--quant-type", "q4_k_m"])
        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(["bash", "scripts/04_quantize_imatrix.sh", "c", "q4_k_m"], check=False)


def test_cli_ui():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["ui"])
        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(["streamlit", "run", "ui/app.py"], check=False)


def test_cli_executable():
    res = subprocess.run(
        [sys.executable, "-m", "maurice.cli", "--version"], capture_output=True, text=True, check=False
    )
    assert res.returncode == 0
    assert f"maurice {__version__}" in res.stdout
