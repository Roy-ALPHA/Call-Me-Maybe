from argparse import ArgumentParser, ArgumentTypeError
import os, sys
from .engine import FunctionCallingEngine

class ArgParser:

    @staticmethod
    def path_extension(path: str) -> str:
        if not path.strip().endswith(".json"):
            raise ArgumentTypeError(f"Invalid path [{path}]: Only JSON files are allowed")
        return path

    @staticmethod
    def validate_path(path: str):
        if not os.path.exists(path):
            raise ArgumentTypeError(f"File not found: {path}")
        return ArgParser.path_extension(path)

    def parse_validate_args(self):
        parser = ArgumentParser(
            prog=f"uv run python -m {os.path.basename(sys.argv[0])}",
            description="These are the available options for configuring the function calling system:"
            )
        parser.add_argument(
            "--functions_definition",
            type=ArgParser.validate_path,
            required=True,
            default=None,
            help="JSON file defining available functions for the LLM"
        )
        parser.add_argument(
            "--input",
            type=ArgParser.validate_path,
            default="data/input/function_calling_tests.json",
            help="input file path"
        )
        parser.add_argument(
            "--output",
            type=ArgParser.path_extension,
            default="data/output/function_calls.json",
            help="output file path"
        )
        return FunctionCallingEngine(args=parser.parse_args())
