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
            "\nDetermine the best matching function name.\n"
            "The answer is: "
        )

        return prompt

    def _build_args_prompt(self, selected_function):

        prompt = (
            self.args_prompt +
            "\nUser Request:\n" + selected_function["prompt"] +
            "\nSelected Function:\n" +
            selected_function["name"] + 
            "\nFunction Parameters (in order):\n" +
            json.dumps(selected_function["parameters"]) +
            "\n"
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

            masked_logits = np.full_like(logits, -np.inf)

            masked_logits[allowed_tokens] = logits[allowed_tokens]

            best_token = np.argmax(masked_logits)

            func.append(best_token)

            cur_node = trie.get_node(best_token, cur_node)

        func_selected = self.model.decode(func)
        for func in self.def_funcs:
            if func["name"] == func_selected[1:]:
                func["prompt"] = cur_prompt
                return self.extract_args(func)
    
    def extract_args(self, func_selected):

        prompt = self._build_args_prompt(func_selected)

        prompt_tokens = self.model.encode(prompt).numpy().ravel().tolist()

        cur_ouput = []

        # open_bracket_tokens =  self.model.encode("(").numpy().ravel().tolist() + self.model.encode(" (").numpy().ravel().tolist()
        # close_bracket_tokens = self.model.encode(")").numpy().ravel().tolist() + self.model.encode(")\n\n").numpy().ravel().tolist() + self.model.encode(")\n").numpy().ravel().tolist()

        for param in func_selected["parameters"]:

            value = []
            can_take = False

            while True:

                logits = np.array(self.model.get_logits_from_input_ids(prompt_tokens + cur_ouput))

                best_token = np.argmax(logits)

                cur_ouput.append(best_token)

                if ")" in self.model.decode(best_token):
                    break

                if can_take:
                    # print(best_token, self.model.decode(best_token))
                    value.append(int(best_token))

                if "(" in self.model.decode(best_token):
                    can_take = True


                # print(best_token, self.model.decode(best_token))

                # logits[best_token] = float("-inf")
            print("".join([self.model.decode(token) for token in value]))


    
    def test(self):
        start = time.perf_counter()
        for prompt in self.inpt_prompts:
            self.extract_func(prompt["prompt"])
        print(time.perf_counter() - start)