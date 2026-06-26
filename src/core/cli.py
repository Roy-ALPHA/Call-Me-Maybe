from argparse import ArgumentParser, ArgumentTypeError, Namespace
import os
import sys
from .engine import FunctionCallingEngine
import json
from .validators import FuncsDef, Prompts
from pydantic import ValidationError


class ArgParser:
    """Parse and validate CLI arguments for the function-calling engine."""

    @staticmethod
    def path_extension(path: str) -> str:
        """Validate that a path points to a JSON file.

        Args:
            path: File path to validate.

        Returns:
            The validated path string.

        Raises:
            ArgumentTypeError: If the path does not end with ``.json``.
        """
        if not path.strip().endswith(".json"):
            msg: str = f"Invalid path [{path}]: Only JSON files are allowed"
            raise ArgumentTypeError(msg)
        return path

    @staticmethod
    def validate_path(path: str) -> str:
        """Validate that a path exists and points to a JSON file.

        Args:
            path: File path to validate.

        Returns:
            The validated path string.

        Raises:
            ArgumentTypeError: If the file does not exist or is not JSON.
        """
        if not os.path.exists(path):
            raise ArgumentTypeError(f"File not found: {path}")
        return ArgParser.path_extension(path)

    @staticmethod
    def validate_input_files(args: Namespace) -> FunctionCallingEngine:
        """Load and validate the function definition and input files.

        Args:
            args: Parsed CLI arguments.

        Returns:
            A configured ``FunctionCallingEngine`` instance.

        Raises:
            ArgumentTypeError: If the input files are invalid or unreadable.
        """
        with open(args.functions_definition) as func_def_f, \
                open(args.input) as input_f:
            try:
                def_funcs = json.load(func_def_f)
                prompts = json.load(input_f)
                FuncsDef(funcs=def_funcs)
                Prompts(prompts=prompts)
                return FunctionCallingEngine(
                    args=args,
                    def_funcs=def_funcs,
                    inpt_prompts=prompts
                )
            except json.JSONDecodeError as e:
                raise ArgumentTypeError(
                    f"Invalid JSON: line {e.lineno}, column {e.colno}: {e.msg}"
                )

            except ValidationError as e:
                msg = str(e).split("For further information visit")[0]
                err_msg = f"Input validation failed:\n{msg}"
                raise ArgumentTypeError(err_msg)

            except OSError as e:
                raise ArgumentTypeError(
                    f"Unable to read input file: {e}"
                )

    def parse_validate_args(self) -> FunctionCallingEngine:
        """Parse CLI arguments and return a configured engine instance.

        Returns:
            A validated ``FunctionCallingEngine`` ready to execute.
        """
        parser = ArgumentParser(
            prog=f"uv run python -m {os.path.basename(sys.argv[0])}",
            description="Available options for configuring the "
                        "function calling system:"
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
        return ArgParser.validate_input_files(parser.parse_args())
