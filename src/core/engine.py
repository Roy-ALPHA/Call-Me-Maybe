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
            self.args_prompt +
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

            best_token = max(allowed_tokens, key=lambda token: logits[token])

            func.append(best_token)

            cur_node = trie.get_node(best_token, cur_node)

        func_selected = self.model.decode(func)

        for func in self.def_funcs:
            if func["name"] == func_selected[1:]:
                func["prompt"] = cur_prompt
                return self.extract_args(func)
    


    def _extract_int_arg(self, tokens):

        num = ""
        generated = []

        while True:

            logits = np.array(self.model.get_logits_from_input_ids(tokens + generated))

            best_token = np.argmax(logits)

            generated.append(best_token)

            text = self.model.decode(best_token)

            number = re.search(r"\d+", text)

            if number:
                num += number.group()
            elif num:
                break

        return int(num)


    def _extract_str_arg(self, tokens, cur_arg):
        
        text = ""
        generated = []

        while True:

            logits = np.array(self.model.get_logits_from_input_ids(tokens + generated))

            best_token = np.argmax(logits)

            text += self.model.decode(best_token)

            




    def _extract_bool_arg(self, tokens):
        pass

    def extract_args(self, func_selected):
        
        prompt_tokens = self.model.encode(func_selected["prompt"]).numpy().ravel().tolist()
        args = dict()

        for arg in func_selected["parameters"]:
            if func_selected["parameters"][arg]["type"] == "number":
                args.update({arg: self._extract_int_arg(prompt_tokens)})
            elif func_selected["parameters"][arg]["type"] == "string":
                args.update({arg: self._extract_str_arg(prompt_tokens, arg)})
            elif func_selected["parameters"][arg]["type"] == "boolean":
                args.update({arg: self._extract_bool_arg(prompt_tokens)})


    
    def test(self):
        start = time.perf_counter()
        for prompt in self.inpt_prompts:
            self.extract_func(prompt["prompt"])
        print((time.perf_counter() - start) / 60)