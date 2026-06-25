from pydantic import BaseModel, ConfigDict
from argparse import Namespace
from llm_sdk import Small_LLM_Model
import numpy as np
import json
import re
import time
from .validators import *
from .Trie import TrieNode, Trie
from multiprocessing import Pool, cpu_count
from functools import cached_property
from pathlib import Path


class FunctionCallingEngine(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    args: Namespace
    model: Small_LLM_Model = Small_LLM_Model()
    def_funcs: list[dict]
    inpt_prompts: list[dict]
    func_prompt: str = ""
    args_prompt: str = ""

    @staticmethod
    def mask_low_scores(allowed_tokens, logits):
        return {token: logits[token] for token in allowed_tokens}

    @cached_property
    def _build_trie(self):
        trie = Trie()

        for func in self.def_funcs:
            trie.insert(
                self.model.encode(" " + func["name"])
                .numpy()
                .ravel()
                .tolist()
            )

        with open("prompts/func_prompt.txt") as f1, open("prompts/args_prompt.txt") as f2:
            self.func_prompt = f1.read()
            self.args_prompt = f2.read()

        return trie

    def _build_func_prompt(self, cur_prompt):

        prompt = (
            self.func_prompt +
            "\nUser Request:\n" + cur_prompt +
            "\nAvailable Functions:\n" +
            json.dumps(self.def_funcs) +
            "\nDetermine the best matching function name.\n" +
            "The output is: "
        )

        return prompt

    def _build_args_prompt(self, selected_function):

        prompt = (
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

    def extract_func(self, cur_prompt):
        
        trie, prompt = (self._build_trie, self._build_func_prompt(cur_prompt))

        func = []

        cur_node = trie.root
        prompt_tokens = self.model.encode(prompt).numpy().ravel().tolist()

        while not cur_node.is_end:

            logits = np.array(self.model.get_logits_from_input_ids(prompt_tokens + func))

            allowed_tokens = trie.get_allowed_next_tokens(cur_node)

            best_token = max(allowed_tokens, key=lambda token: logits[token])

            func.append(best_token)

            cur_node = trie.get_node(best_token, cur_node)

        func_selected = self.model.decode(func)

        for func in self.def_funcs:
            if func["name"] == func_selected[1:]:
                func["prompt"] = cur_prompt
                return self.extract_args(func)


    def extract_args(self, func_selected):

        allowed_numbers = set()
        for i in range(10):
            allowed_numbers.update(
                self.model.encode(str(i)).numpy().ravel().tolist()
            )
        allowed_numbers.update(self.model.encode(" -").numpy().ravel().tolist())
        allowed_numbers.update(self.model.encode(",").numpy().ravel().tolist())
        allowed_numbers.update(self.model.encode(".").numpy().ravel().tolist())
        allowed_numbers.update(self.model.encode("\n").numpy().ravel().tolist())

        trie = Trie()
        trie.insert(self.model.encode("true").numpy().ravel().tolist())
        trie.insert(self.model.encode("false").numpy().ravel().tolist())

        prompt_text = (
            self._build_args_prompt(func_selected)
        )

        final_res = {
            "prompt": func_selected["prompt"],
            "name": func_selected["name"],
            "parameters": dict()
        }

        for arg in func_selected["parameters"]:

            generated = []
            arg_type = func_selected["parameters"][arg]["type"]
            prompt_text += f"\n{arg}="
            cur_node = trie.root


            prompt_tokens = (
                self.model.encode(prompt_text)
                .numpy()
                .ravel()
                .tolist()
            )

            while True:

                logits = np.array(
                    self.model.get_logits_from_input_ids(
                        prompt_tokens + generated
                    )
                )

                if arg_type == "number":

                    best_token = max(
                        allowed_numbers,
                        key=lambda t: logits[t]
                    )

                    generated.append(best_token)

                    tmp_text = self.model.decode(best_token)
                    if "," in tmp_text or "\n" in tmp_text:
                        final_res["parameters"].update({arg: float(self.model.decode(generated).rstrip(","))})
                        break 

                if arg_type == "boolean":

                    allowed_tokens = trie.get_allowed_next_tokens(cur_node)

                    best_token = max(
                        allowed_tokens,
                        key=lambda t: logits[t]
                    )

                    generated.append(best_token)

                    cur_node = trie.get_node(best_token, cur_node)

                    if cur_node.is_end:
                        final_res["parameters"].update({arg: self.model.decode(generated).strip() == "true"})
                        break
                
                if arg_type == "string":

                    best_token = np.argmax(logits)

                    generated.append(best_token)

                    if "\n" in self.model.decode(best_token):
                        final_res["parameters"].update({arg: self.model.decode(generated).strip().strip('"')})
                        break

            value = self.model.decode(generated)

            prompt_text += value

        return final_res

    def call_me_maybe(self):
        output = []
        start = time.perf_counter()
        for prompt in self.inpt_prompts:
            output.append(self.extract_func(prompt["prompt"]))

        output_path = Path(self.args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(output, f, indent=4)
        
        print("Time of execution:", round(((time.perf_counter() - start) / 60), 2), "min")
