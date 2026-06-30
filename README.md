*This project has been created as part of the 42 curriculum by yoelboud.*

Description
-----------
Call-Me-Maybe is a small function-calling system that demonstrates how a causal language model can be used to (1) select the best-matching function for a user's natural-language request, and (2) extract typed arguments for that function. The project implements a constrained-decoding approach where allowed next tokens are restricted using a Trie built from tokenized function names. It is intended for educational use and experimentation with local Hugging Face models.

Instructions
------------
- **Prerequisites**: Python 3.10+, `git`, and network access to download Hugging Face models.

Example setup with `uv` and `make`:

```bash
make install
```

- **Run the program** (example):

```bash
make run ARGS="--functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json"
```

By default, the example inputs are in `data/input/` and functions are defined in `data/input/functions_definition.json`.

Useful `make` targets:

```bash
make install
make run ARGS="--functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json"
make debug ARGS="--functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json"
make lint
make clean
```

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
```
flowchart TD

subgraph group_io["CLI / I/O"]
  node_main(("__main__<br/>entrypoint<br/>[__main__.py]"))
  node_cli["CLI<br/>parser<br/>[cli.py]"]
  node_writer["Result write<br/>output"]
end

subgraph group_core["Core engine"]
  node_validators["Validators<br/>schema gate<br/>[validators.py]"]
  node_engine["Engine<br/>orchestrator<br/>[engine.py]"]
  node_trie{{"Trie<br/>constraint index<br/>[Trie.py]"}}
  node_func_select(("Func select<br/>phase"))
  node_arg_extract(("Arg extract<br/>phase"))
  node_constrained_decode{{"Constrained decode<br/>decoder"}}
  node_prompt_build["Prompt build<br/>template step"]
end

subgraph group_runtime["Local model"]
  node_llm_sdk["LLM SDK<br/>model boundary<br/>[__init__.py]"]
  node_tokenizer(("Tokenizer<br/>token-id alignment"))
  node_logits(("Logits<br/>scoring"))
end

subgraph group_assets["Assets"]
  node_func_prompt["Function prompt<br/>template<br/>[func_prompt.txt]"]
  node_function_defs["Function catalog<br/>input json"]
  node_test_prompts["Test prompts<br/>input json"]
end

node_main -->|"start"| node_cli
node_cli -->|"validate"| node_validators
node_cli -->|"run"| node_engine
node_function_defs -->|"schema"| node_validators
node_test_prompts -->|"schema"| node_validators
node_validators -->|"clean inputs"| node_engine
node_engine -->|"assemble"| node_prompt_build
node_func_prompt -->|"template"| node_prompt_build
node_function_defs -->|"tokenize"| node_trie
node_trie -->|"allow tokens"| node_func_select
node_prompt_build -->|"selection prompt"| node_func_select
node_func_select -->|"decode"| node_constrained_decode
node_constrained_decode -->|"scores"| node_llm_sdk
node_llm_sdk -->|"token ids"| node_tokenizer
node_tokenizer -->|"align"| node_trie
node_func_select -->|"selected function"| node_arg_extract
node_prompt_build -->|"arg prompt"| node_arg_extract
node_arg_extract -->|"type constraints"| node_constrained_decode
node_constrained_decode -->|"score"| node_logits
node_engine -->|"emit"| node_writer
node_writer -.->|"done"| node_main

click node_main "https://github.com/roy-alpha/call-me-maybe/blob/main/src/__main__.py"
click node_cli "https://github.com/roy-alpha/call-me-maybe/blob/main/src/core/cli.py"
click node_validators "https://github.com/roy-alpha/call-me-maybe/blob/main/src/core/validators.py"
click node_engine "https://github.com/roy-alpha/call-me-maybe/blob/main/src/core/engine.py"
click node_trie "https://github.com/roy-alpha/call-me-maybe/blob/main/src/core/Trie.py"
click node_llm_sdk "https://github.com/roy-alpha/call-me-maybe/blob/main/llm_sdk/__init__.py"
click node_func_prompt "https://github.com/roy-alpha/call-me-maybe/blob/main/prompts/func_prompt.txt"
click node_function_defs "https://github.com/roy-alpha/call-me-maybe/blob/main/data/input/functions_definition.json"
click node_test_prompts "https://github.com/roy-alpha/call-me-maybe/blob/main/data/input/function_calling_tests.json"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_main,node_cli,node_writer toneBlue
class node_validators,node_engine,node_trie,node_func_select,node_arg_extract,node_constrained_decode,node_prompt_build toneAmber
class node_llm_sdk,node_tokenizer,node_logits toneMint
class node_func_prompt,node_function_defs,node_test_prompts toneRose
```
