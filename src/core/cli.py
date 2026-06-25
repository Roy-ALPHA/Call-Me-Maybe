from argparse import ArgumentParser, ArgumentTypeError
import os, sys
from .engine import FunctionCallingEngine
import json
from .validators import *
from pydantic import ValidationError

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

    @staticmethod
    def validate_input_files(args):
        with open(args.functions_definition) as func_def_f, open(args.input) as input_f:
            try:
                def_funcs = json.load(func_def_f)
                prompts = json.load(input_f)
                FuncsDef(funcs=def_funcs)
                Prompts(prompts=prompts)
                return FunctionCallingEngine(args=args, def_funcs=def_funcs, inpt_prompts=prompts)
            except json.JSONDecodeError as e:
                raise ArgumentTypeError(
                    f"Invalid JSON: line {e.lineno}, column {e.colno}: {e.msg}"
                )

            except ValidationError as e:
                msg = str(e).split("For further information visit")[0]
                raise ArgumentTypeError(
                    f"Input validation failed:\n{msg}"
                )

            except OSError as e:
                raise ArgumentTypeError(
                    f"Unable to read input file: {e}"
                )

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
        return ArgParser.validate_input_files(parser.parse_args())
