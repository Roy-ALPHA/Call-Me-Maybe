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
    base_prompt: str = ""

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

        with open("src/core/prompt.txt") as f:
            self.base_prompt = f.read()

        return trie

    def _build_prompt(self, cur_prompt):

        prompt = (
            self.base_prompt +
            "\nUser Request:\n" + cur_prompt +
            "\nAvailable Functions:\n" +
            json.dumps(self.def_funcs) +
            "\nDetermine the best matching function name.\n"
            "The answer is: "
        )

        return prompt


    def extract_func(self, cur_prompt):
        
        trie, prompt = (self._build_trie, self._build_prompt(cur_prompt))

        func = []

        cur_node = trie.root
        while not cur_node.is_end:
            prompt_tokens = self.model.encode(prompt).numpy().ravel().tolist()

            logits = np.array(self.model.get_logits_from_input_ids(prompt_tokens + func))

            allowed_tokens = trie.get_allowed_next_tokens(cur_node)

            masked_logits = np.full_like(logits, -np.inf)

            masked_logits[allowed_tokens] = logits[allowed_tokens]

            best_token = np.argmax(masked_logits)

            func.append(best_token)

            cur_node = trie.get_node(best_token, cur_node)

        return {"prompt": cur_prompt, "func_name": self.model.decode(func)}
    
    def test(self):
        start = time.perf_counter()
        for prompt in self.inpt_prompts:
            print(self.extract_func(prompt["prompt"]))
        print(time.perf_counter() - start)