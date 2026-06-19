from pydantic import BaseModel, ConfigDict
from argparse import Namespace
from llm_sdk import Small_LLM_Model
import numpy as np
import json
import re
import time
from .validators import *
from .Trie import TrieNode, Trie


class FunctionCallingEngine(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    args: Namespace
    model: Small_LLM_Model = Small_LLM_Model()
    def_funcs: list[dict]
    inpt_prompts: list[dict]

    @staticmethod
    def mask_low_scores(allowed_tokens, logits):
        return {token: logits[token] for token in allowed_tokens}

    def extract_func(self, cur_promt):
        print(cur_promt)
        prompt = "prompt: " + f"{cur_promt}" + "\nfrom this prompt, select the best matching dict from this list of dicts:\n" + self.def_funcs.__repr__()

        trie = Trie()
        for func in self.def_funcs:
            trie.insert(self.model.encode(" " + func.__repr__() + "}").numpy().ravel().tolist())

        generated_tokens = []
        correct_func = ""

        cur_node = trie.root
        while not cur_node.is_end:
            prompt_tokens = self.model.encode(prompt).numpy().ravel().tolist()

            logits = self.model.get_logits_from_input_ids(prompt_tokens + generated_tokens)

            top_k = np.argmax(np.array(logits))

            allowed_tokens = trie.get_allowed_next_tokens(cur_node)

            generated_tokens.append(top_k)

            if top_k in allowed_tokens:
                valid_tokens = FunctionCallingEngine.mask_low_scores(allowed_tokens, logits)

                best_token = max(valid_tokens, key=valid_tokens.get)

                correct_func += self.model.decode(best_token)
                print(correct_func)
                cur_node = trie.get_node(best_token, cur_node)

        print(json.loads(json.dumps(correct_func)))
    
    def test(self):
        for prompt in self.inpt_prompts:
            self.extract_func(prompt["prompt"])

