"""Command-line entry point for NuSA-CL experiments."""

from .args import parse_arguments
from .trainer import train


def main(args):
    return train(args)


if __name__ == "__main__":
    main(parse_arguments())
