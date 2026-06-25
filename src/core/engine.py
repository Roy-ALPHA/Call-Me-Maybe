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
            "I will give you the Parameter name and you give me hes value (add ',' in the end of value)."
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
                return self.extract_test(func)

    def _extract_int_arg(self, tokens: list, cur_arg) -> int:
        """
        Greedily decodes tokens until a complete integer is found.
        Accumulates digit characters from each decoded token and stops
        once a non-digit token is encountered after digits have been collected.
        """
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

    def _extract_str_arg(self, tokens: list, cur_arg: str):
        """
        Greedily decodes tokens and accumulates text, attempting to parse
        it as JSON after each token. Returns the value at `cur_arg` once
        a valid JSON object containing that key is produced.
        """
        text = ""
        decoder = json.JSONDecoder()
        generated = []
 
        while True:
            logits = np.array(self.model.get_logits_from_input_ids(tokens + generated))
            best_token = np.argmax(logits)
            generated.append(best_token)
            text += self.model.decode(best_token)

            try:
                json_obj, _ = decoder.raw_decode(text)
                if cur_arg in json_obj:
                    return json_obj[cur_arg]
            except json.JSONDecodeError:
                pass
 
    def _extract_bool_arg(self, tokens: list) -> bool:
        """
        Uses a Trie of valid boolean token sequences ("true" / "false") to
        constrain generation. At each step, only tokens that are valid
        continuations in the Trie are considered, picking the highest-logit
        one. Guarantees the output is always exactly "true" or "false".
        """
        trie = Trie()
        trie.insert(self.model.encode("true").numpy().ravel().tolist())
        trie.insert(self.model.encode("false").numpy().ravel().tolist())
 
        generated = []
        cur_node = trie.root
 
        while not cur_node.is_end:
            logits = np.array(self.model.get_logits_from_input_ids(tokens + generated))
            allowed_tokens = trie.get_allowed_next_tokens(cur_node)
            best_token = max(allowed_tokens, key=lambda token: logits[token])
            generated.append(best_token)
            cur_node = trie.get_node(best_token, cur_node)
 
        return self.model.decode(generated).strip() == "true"


    def extract_test(self, func_selected):

        stop_tokens = self.model.encode(",").numpy().ravel().tolist() + self.model.encode("}").numpy().ravel().tolist()

        allowed_numbers = set()
        for i in range(10):
            allowed_numbers.update(
                self.model.encode(str(i)).numpy().ravel().tolist()
            )
        allowed_numbers.update(self.model.encode("-").numpy().ravel().tolist())
        allowed_numbers.update(self.model.encode(",").numpy().ravel().tolist())
        allowed_numbers.update(self.model.encode("}").numpy().ravel().tolist())

        trie = Trie()
        trie.insert(self.model.encode("true").numpy().ravel().tolist())
        trie.insert(self.model.encode("false").numpy().ravel().tolist())

        # generated = []
        # prompt_tokens = self.model.encode(self._build_args_prompt(func_selected)).numpy().ravel().tolist()

        prompt_text = (
            self._build_args_prompt(func_selected)
        )

        prompt_tokens = self.model.encode(
            prompt_text
        ).numpy().ravel().tolist()


        for arg in func_selected["parameters"]:

            generated = []
            arg_prompt = prompt_tokens + self.model.encode(f"\n{arg}=").numpy().ravel().tolist()

            while True:

                logits = np.array(
                    self.model.get_logits_from_input_ids(
                        arg_prompt + generated
                    )
                )

                best_token = max(
                    allowed_numbers,
                    key=lambda t: logits[t]
                )
                print(self.model.decode(arg_prompt))
                generated.append(best_token)

                if best_token in stop_tokens:
                    break

            number_text = self.model.decode(generated)
            print(arg, number_text)





    def extract_args(self, func_selected):
        
        prompt_tokens = self.model.encode(self._build_args_prompt(func_selected)).numpy().ravel().tolist()
        args = dict()

        for arg in func_selected["parameters"]:
            if func_selected["parameters"][arg]["type"] == "number":
                args.update({arg: self._extract_int_arg(prompt_tokens, arg)})
            elif func_selected["parameters"][arg]["type"] == "string":
                args.update({arg: self._extract_str_arg(prompt_tokens, arg)})
            elif func_selected["parameters"][arg]["type"] == "boolean":
                args.update({arg: self._extract_bool_arg(prompt_tokens)})
        
        print(args)


    
    def test(self):
        start = time.perf_counter()
        for prompt in self.inpt_prompts:
            self.extract_func(prompt["prompt"])
        print((time.perf_counter() - start) / 60)