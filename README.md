*This project has been created as part of the 42 curriculum by yoelboud.*

Description
-----------
Call-Me-Maybe is a small function-calling system that demonstrates how a causal language model can be used to (1) select the best-matching function for a user's natural-language request, and (2) extract typed arguments for that function. The project implements a constrained-decoding approach where allowed next tokens are restricted using a Trie built from tokenized function names. It is intended for educational use and experimentation with local Hugging Face models.

Instructions
------------
- **Prerequisites**: Python 3.10+, git, network access to download Hugging Face models. Recommended creating a virtual environment.

Example setup (bash / zsh):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

- **Run the program** (example):

```bash
python -m src --functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calls.json
```

By default the example inputs are in `data/input/` and functions are defined in `data/input/functions_definition.json`.

Algorithm explanation
---------------------
This project uses a constrained decoding pipeline to ensure the language model can only produce one of the provided function names when selecting a function, and to guide argument generation by token-level constraints.

High-level steps:
- Build a Trie of token ID sequences representing each function name (encoded using the model tokenizer).
- Create a function-selection prompt (from `prompts/func_prompt.txt`) that includes the user's request and the JSON list of available functions.
- Encode the prompt and then, at generation time, repeatedly query the model's logits for the next token, but only allow tokens that appear in the Trie node for the current prefix. Choose the token with the highest logit among allowed tokens (greedy constrained decoding). When a terminal node is reached, decode the chosen token sequence to a function name.
- For argument extraction, the engine builds a second prompt describing the selected function, its parameters and types, then generates each parameter value. Numeric arguments use a restricted token set of digits, signs and separators; booleans are constrained to `true`/`false` tokens via a small trie; strings are greedily decoded until a newline delimiter is produced.

Design decisions
----------------
- **Trie-based constrained decoding**: Simple, deterministic, and effective for forcing the model to pick exactly one of the available function names.
- **Greedy selection using raw logits**: Simpler and faster than beam search; appropriate here since the Trie already constrains outputs to valid options.
- **Pydantic for input validation**: `src/core/validators.py` ensures the functions-definition and prompts follow the expected schema.
- **Prompt templates**: Separated into `prompts/func_prompt.txt` and generated argument prompts to keep system instruction and user prompt content modular.

Performance analysis
--------------------
- **Accuracy**: Heavily depends on the underlying model. Constrained decoding eliminates many incorrect function-name outputs, improving selection precision for cases where function names are unambiguous. Argument extraction accuracy depends on tokenization and prompt clarity.
- **Reliability**: The Trie constraint reduces incorrect selections but requires exact tokenization alignment between function name encodings and the model tokenizer. Edge cases (ambiguous names, overlapping token sequences) can still cause issues.

Challenges faced
----------------
- Handling tokenization boundaries for numbers and punctuation when extracting numeric arguments.
- Ensuring boolean values are generated reliably (implemented a small Trie for `true`/`false`).
- Robustness of prompt engineering: designing prompts to produce consistent, machine-parsable outputs.

Testing strategy
----------------
- Input validation: `src/core/validators.py` uses Pydantic models to check function definitions and prompts before running.
- Example dataset: `data/input/function_calling_tests.json` contains representative prompts used for manual/automated checks.
- Manual integration runs: use the example run command and inspect `data/output/function_calls.json` for expected function names and parsed argument types.

Example usage
-------------
Command:

```bash
python -m src --functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calls.json
```

Expected output (example snippet) — `data/output/function_calls.json`:

```json
[
	{
		"prompt": "What is the div of -2 and -3?",
		"name": "fn_add_numbers",
		"parameters": {"a": -2, "b": -3}
	},
	{
		"prompt": "Greet shrek",
		"name": "fn_greet",
		"parameters": {"name": "shrek"}
	}
]
```

Resources
---------
- Pydantic documentation: https://docs.pydantic.dev/
- Trie and constrained decoding related reading: research on constrained and lexically-constrained decoding (e.g., "Lexically constrained decoding for neural machine translation").

Use of AI
---------
AI was used to support the implementation and improve the overall design.

Specifically, AI was used for:

- Brainstorming the architecture of the function-calling engine.
- Reviewing and improving prompts used for function selection, argument extraction, and function validation.
- Discussing constrained decoding techniques, including Trie-based decoding, and token constraints.
- Explaining Python concepts, regular expressions, tokenization, and algorithmic complexity.
- Assisting with debugging, refactoring, and improving code readability.
- Suggesting improvements for error handling, input validation, and README documentation.

Notes and next steps
--------------------
- For production or large-scale usage consider using batched calls, caching, beam search when needed, and more robust token-to-value parsers.
- Add unit tests and CI, include a lightweight mocked model for deterministic tests.

Credits
-------
Project scaffold and code authored as part of the 42 curriculum.
