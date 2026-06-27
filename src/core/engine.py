from pydantic import BaseModel, ConfigDict
from argparse import Namespace
from llm_sdk import Small_LLM_Model
import numpy as np
import json
import time
from .Trie import Trie, TrieNode
from functools import cached_property
from pathlib import Path


class FunctionCallingEngine(BaseModel):
    """Run constrained function selection and argument extraction.

    The engine loads the available function definitions and user prompts,
    selects the best matching function with constrained decoding, and then
    extracts typed arguments for the selected function.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    args: Namespace
    model: Small_LLM_Model = Small_LLM_Model()
    def_funcs: list[dict]
    inpt_prompts: list[dict]
    func_prompt: str = ""
    args_prompt: str = ""

    @cached_property
    def _build_trie(self) -> Trie:
        """Build a trie containing tokenized function names.

        Returns:
            A ``Trie`` that constrains generation to valid function names.

        Raises:
            RuntimeError: If the prompt file cannot be loaded or the trie
                cannot be constructed.
        """
        try:
            trie: Trie = Trie()

            for func in self.def_funcs:
                trie.insert(
                    self.model.encode(" " + func["name"])
                    .numpy()
                    .ravel()
                    .tolist()
                )

            with open("prompts/func_prompt.txt") as f1:
                self.func_prompt = f1.read()

            return trie
        except FileNotFoundError as e:
            raise RuntimeError(f"Prompt file not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to build Trie: {e}")

    def _build_func_prompt(self, cur_prompt: str) -> str:
        """Build the prompt used to select the best matching function.

        Args:
            cur_prompt: The current user request.

        Returns:
            The full prompt string passed to the model.
        """

        prompt: str = (
            self.func_prompt +
            "\nUser Request:\n" + cur_prompt +
            "\nAvailable Functions:\n" +
            json.dumps(self.def_funcs) +
            "\nDetermine the best matching function name.\n" +
            "The output is: "
        )

        return prompt

    def _build_args_prompt(self, selected_function: dict) -> str:
        """Build the prompt used to extract arguments for a function.

        Args:
            selected_function: The selected function definition augmented with
                the original user prompt.

        Returns:
            The full prompt string used for argument extraction.
        """

        prompt: str = (
            "\nUser Request:\n" + selected_function["prompt"] +
            "\nSelected Function:\n" +
            selected_function["name"] +
            "\nFunction Parameters (in order):\n" +
            json.dumps(selected_function["parameters"]) +
            "\nFunction description:\n" +
            selected_function["description"] + "\n" +
            'if Parameter type is "boolean" return (true or false).' +
            "\nI will give you the Parameter name and you give me hes value.\n"
        )

        return prompt

    def extract_func(self, cur_prompt: str) -> dict:
        """Select the best matching function and extract its arguments.

        Args:
            cur_prompt: The natural-language request from the user.

        Returns:
            A dictionary containing the selected function name, prompt, and
            parsed parameters.

        Raises:
            RuntimeError: If function extraction fails.
        """
        try:
            trie: Trie = self._build_trie
            prompt: str = self._build_func_prompt(cur_prompt)

            func_tokens: list[int] = []

            cur_node: TrieNode = trie.root
            prompt_tokens: list[int] = (
                self.model.encode(prompt).numpy().ravel().tolist()
            )

            while not cur_node.is_end:

                logits: list[float] = self.model.get_logits_from_input_ids(
                    prompt_tokens + func_tokens
                )

                allowed_tokens: list[int] = trie.get_allowed_next_tokens(
                    cur_node
                )

                best_token: int = max(
                    allowed_tokens, key=lambda token: logits[token]
                )

                func_tokens.append(best_token)

                cur_node = trie.get_node(best_token, cur_node)

            func_selected: str = self.model.decode(func_tokens)

            for func_def in self.def_funcs:
                if func_def["name"] == func_selected[1:]:
                    func_def["prompt"] = cur_prompt
                    return self.extract_args(func_def)

            raise RuntimeError("No matching function found")
        except Exception as e:
            raise RuntimeError(f"Error extracting function: {e}")

    def extract_args(self, func_selected: dict) -> dict:
        """Extract typed arguments for the selected function.

        Args:
            func_selected: The selected function definition plus the original
                prompt.

        Returns:
            A dictionary containing the prompt, function name, and parsed
            parameter values.
        """

        allowed_numbers: set[int] = set()
        for i in range(10):
            allowed_numbers.update(
                self.model.encode(str(i)).numpy().ravel().tolist()
            )
        allowed_numbers.update(
            self.model.encode(" -").numpy().ravel().tolist()
        )
        allowed_numbers.update(
            self.model.encode(",").numpy().ravel().tolist()
        )
        allowed_numbers.update(
            self.model.encode(".").numpy().ravel().tolist()
        )
        allowed_numbers.update(
            self.model.encode("\n").numpy().ravel().tolist()
        )

        trie: Trie = Trie()
        trie.insert(self.model.encode("true").numpy().ravel().tolist())
        trie.insert(self.model.encode("false").numpy().ravel().tolist())
        trie.insert(self.model.encode(" true").numpy().ravel().tolist())
        trie.insert(self.model.encode(" false").numpy().ravel().tolist())

        prompt_text: str = (
            self._build_args_prompt(func_selected)
        )

        final_res: dict = {
            "prompt": func_selected["prompt"],
            "name": func_selected["name"],
            "parameters": dict()
        }

        for arg in func_selected["parameters"]:

            generated: list[int] = []
            arg_type: str = func_selected["parameters"][arg]["type"]
            prompt_text += f"\n{arg}="
            cur_node: TrieNode = trie.root

            prompt_tokens: list[int] = (
                self.model.encode(prompt_text)
                .numpy()
                .ravel()
                .tolist()
            )

            while True:

                logits: list[float] = self.model.get_logits_from_input_ids(
                    prompt_tokens + generated
                )

                if arg_type in ["number", "integer"]:

                    best_token: int = max(
                        allowed_numbers,
                        key=lambda t: logits[t]
                    )

                    generated.append(best_token)

                    tmp_text: str = self.model.decode([best_token])
                    if "," in tmp_text or "\n" in tmp_text:
                        if arg_type == "number":
                            decoded_value = self.model.decode(generated)
                            decoded_value = decoded_value.rstrip(",")
                            final_res["parameters"].update(
                                {arg: float(decoded_value)}
                            )
                        else:
                            value = self.model.decode(generated)
                            value = value.rstrip(",").split(".")[0]
                            final_res["parameters"].update(
                                {arg: int(value)}
                            )
                        break

                if arg_type == "boolean":

                    allowed_tokens = trie.get_allowed_next_tokens(
                        cur_node
                    )

                    bool_token: int = max(
                        allowed_tokens,
                        key=lambda t: logits[t]
                    )

                    generated.append(bool_token)

                    cur_node = trie.get_node(bool_token, cur_node)

                    if cur_node.is_end:
                        value = self.model.decode(generated).strip()
                        final_res["parameters"].update(
                            {arg: value == "true"}
                        )
                        break

                if arg_type == "string":

                    text_token = int(np.argmax(logits))

                    generated.append(text_token)

                    if "\n" in self.model.decode([text_token]):
                        value = self.model.decode(generated).strip()
                        final_res["parameters"].update(
                            {arg: value.strip('"')}
                        )
                        break

            value = self.model.decode(generated)

            prompt_text += value

        return final_res

    def call_me_maybe(self) -> None:
        """Process all prompts and write the extracted calls to disk.

        Raises:
            RuntimeError: If writing the output file or processing prompts
                fails.
        """
        try:
            output: list[dict] = []
            start = time.perf_counter()
            for prompt in self.inpt_prompts:
                try:
                    output.append(self.extract_func(prompt["prompt"]))
                except Exception as e:
                    print(f"Warning: Failed to process prompt: {e}")
                    continue

            output_path = Path(self.args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w") as f:
                json.dump(output, f, indent=4)

            print(
                "Time of execution:",
                round(((time.perf_counter() - start) / 60), 2),
                "min"
            )
        except IOError as e:
            raise RuntimeError(f"Failed to write output: {e}")
        except Exception as e:
            raise RuntimeError(f"Error during processing: {e}")
