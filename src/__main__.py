from .core import ArgParser
from argparse import ArgumentTypeError


def run() -> None:
    """Run the command-line interface for the function-calling engine."""
    try:
        engine = ArgParser().parse_validate_args()
        engine.call_me_maybe()
    except ArgumentTypeError as e:
        print(f"Error: {e}")
        exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    run()
