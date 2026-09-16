"""Command-line arguments for the MTIL release code."""

from __future__ import annotations

import argparse

import torch


def _csv(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def parse_arguments():
    parser = argparse.ArgumentParser(description="NuSA-CL continual adaptation")
    parser.add_argument(
        "--method",
        choices=("full_ft", "lora", "milora", "inflora", "nusa"),
        default="nusa",
    )
    parser.add_argument("--model", default="ViT-B/16")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--data-location", default="./data")
    parser.add_argument("--train-dataset", "--train_dataset", default="Aircraft")
    parser.add_argument("--eval-datasets", type=_csv, default=None)
    parser.add_argument("--template", default=None)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-size-eval", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--few-shot", "--few_shot", type=int, default=None)
    steps = parser.add_mutually_exclusive_group()
    steps.add_argument("--iterations", type=int, default=None)
    steps.add_argument("--epochs", type=int, default=None)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--label-smoothing", "--ls", type=float, default=0.2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)

    parser.add_argument("--encoder", choices=("text", "vision", "both"), default="both")
    parser.add_argument(
        "--position",
        choices=("bottom", "mid", "up", "half-up", "half-bottom", "top3", "all"),
        default="all",
    )
    parser.add_argument("--params", nargs="+", choices=("q", "k", "v", "o"), default=["q", "k", "v", "o"])
    parser.add_argument("--params-text", nargs="+", choices=("q", "k", "v", "o"), default=None)
    parser.add_argument("--params-vision", nargs="+", choices=("q", "k", "v", "o"), default=None)
    parser.add_argument("--rank", "--r", type=int, default=128)
    parser.add_argument("--r-text", "--r_text", type=int, default=None)
    parser.add_argument("--r-vision", "--r_vision", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--dropout", "--dropout_rate", type=float, default=0.25)
    parser.add_argument("--cutoff", type=float, default=0.95)
    parser.add_argument("--gpm-threshold", type=float, default=0.95)

    parser.add_argument("--save", default="output/nusa")
    parser.add_argument("--load", default=None)

    args = parser.parse_args()
    if args.iterations is None and args.epochs is None:
        args.iterations = 1000
    if not 0.0 < args.cutoff < 1.0:
        parser.error("--cutoff must lie strictly between 0 and 1")
    if not 0.0 < args.gpm_threshold < 1.0:
        parser.error("--gpm-threshold must lie strictly between 0 and 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup-ratio must lie in [0, 1)")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    args.backbone = args.backbone or args.model
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args
