"""
maurice CLI Entrypoint (maurice/cli.py)

Unifies the pipeline into a single `maurice` command with subcommands.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="maurice",
        description="Minimal Adaptation for Ultra-fast Reasoning and Inference in Code Engines",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    subparsers.required = True

    # maurice prepare
    parser_prep = subparsers.add_parser("prepare", help="Prepare datasets")
    parser_prep.add_argument("--variant", choices=["c", "r", "g", "all"], default="c", help="Target variant")
    parser_prep.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")

    # maurice train
    parser_train = subparsers.add_parser("train", help="Run QLoRA training")
    parser_train.add_argument("--variant", choices=["c", "r", "g"], required=True, help="Target variant")
    parser_train.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")

    # maurice merge
    parser_merge = subparsers.add_parser("merge", help="Merge adapter weights")
    parser_merge.add_argument("--variant", choices=["c", "r", "g"], required=True, help="Target variant")
    parser_merge.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")

    # maurice quantize
    parser_quantize = subparsers.add_parser("quantize", help="Quantize to GGUF using imatrix")
    parser_quantize.add_argument("--variant", choices=["c", "r", "g"], required=True, help="Target variant")

    # maurice eval
    parser_eval = subparsers.add_parser("eval", help="Run benchmark evaluation")
    parser_eval.add_argument("--variant", choices=["c", "r", "g", "all"], default="all", help="Target variant")
    parser_eval.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")

    # maurice serve
    parser_serve = subparsers.add_parser("serve", help="Start FastAPI inference server")
    parser_serve.add_argument("--variant", choices=["c", "r", "g"], default="c", help="Target variant")
    parser_serve.add_argument("--port", type=int, default=8000, help="Server port")
    parser_serve.add_argument("--engine", choices=["hf", "vllm", "mock"], default="hf", help="Inference engine backend")

    # maurice ui
    subparsers.add_parser("ui", help="Launch Streamlit UI")

    # maurice synth-prefs
    parser_synth = subparsers.add_parser("synth-prefs", help="Synthesize preference datasets using RLAIF")
    parser_synth.add_argument("--variant", choices=["c", "r", "g"], required=True, help="Model variant")
    parser_synth.add_argument("--input", type=str, help="Input SFT dataset")
    parser_synth.add_argument("--output", type=str, help="Output preference dataset")
    parser_synth.add_argument("--samples", type=int, default=-1, help="Number of samples to process")
    parser_synth.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    # maurice align
    parser_align = subparsers.add_parser("align", help="Run Post-SFT alignment (DPO/ORPO)")
    parser_align.add_argument("--variant", choices=["c", "r", "g"], required=True, help="Target variant")
    parser_align.add_argument("--method", choices=["dpo", "orpo"], default="orpo", help="Alignment method")
    parser_align.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")

    args = parser.parse_args()

    if args.command == "prepare":
        from maurice.prepare import main as prep_main

        prep_main(sys.argv[2:])
    elif args.command == "train":
        from maurice.train import main as train_main

        train_main(sys.argv[2:])
    elif args.command == "merge":
        from maurice.merge import main as merge_main

        merge_main(sys.argv[2:])
    elif args.command == "eval":
        from maurice.eval import main as eval_main

        eval_main(sys.argv[2:])
    elif args.command == "quantize":
        print(f"Quantizing variant {args.variant} (delegating to shell script)...")
        import subprocess

        subprocess.run(["bash", "scripts/04_quantize_imatrix.sh", args.variant], check=False)
    elif args.command == "serve":
        from maurice.serve import main as serve_main

        serve_main(sys.argv[2:])
    elif args.command == "ui":
        import subprocess

        subprocess.run(["streamlit", "run", "ui/app.py"], check=False)
    elif args.command == "synth-prefs":
        from maurice.synth import main as synth_main

        synth_main(sys.argv[2:])
    elif args.command == "align":
        from maurice.align import main as align_main

        align_main(sys.argv[2:])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
