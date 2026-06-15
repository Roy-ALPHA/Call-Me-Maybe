from pydantic import BaseModel, ConfigDict
from argparse import Namespace
from llm_sdk import Small_LLM_Model
import numpy as np
import json
import re
import time


class FunctionCallingEngine(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    args: Namespace
    model: Small_LLM_Model = Small_LLM_Model()
    def_funcs: dict = None
    inpt_prompts: dict = None

    def _encode_func_names(self) -> dict:
        encoded_funcs = dict()
        with open(self.args.functions_definition) as f:
            self.def_funcs = json.load(f)
        
        index = 0
        for func in self.def_funcs:
            tokens_IDs = self.model.encode(f"{func['description']} {func['name']}").numpy().ravel()
            encoded_funcs.update({str(index): tokens_IDs})
            index += 1

        return encoded_funcs

    @staticmethod
    def _check_if_found(predictable_paths: list):
        tmp = list()
        test = list()
        for path in predictable_paths:
            if len(path["IDs"]) == path["elem_founded"]:
                if max(predictable_paths, key=lambda path: path["scores"]) == path:
                    test.append(path["IDs"])
                    tmp.append(path)
        if not test:
            return False
        for p in tmp:
            predictable_paths.remove(p)
        return test

    def _get_logits(self, text: str):
        tensors = self.model.encode(text).numpy()
        return np.array(self.model.get_logits_from_input_ids(tensors.ravel().tolist()))

    def func_names_filter(self, prompt_logits: np) -> list[int]:
        encoded_funcs = self._encode_func_names()

        scores = {}
        for idx, token_ids in encoded_funcs.items():
            token_ids = np.array(token_ids)
            scores[idx] = np.sum(prompt_logits[token_ids])


        best_idx = max(scores, key=scores.get)
        return  self.def_funcs[int(best_idx)]

    def json_gen(self, json_out: str, prompt: int):
        prompt_text = self.inpt_prompts[prompt].get("prompt")
        func_name = self.func_names_filter(self._get_logits(prompt_text))["name"]
        parameters = "test"
        prompt_json = json.dumps(prompt_text)
        name_json = json.dumps(func_name)
        parameters_json = json.dumps(parameters)
        commas = "}" if prompt == len(self.inpt_prompts) - 1 else "},"
        stats = {
            "state1": self.model.encode("[").numpy().ravel(),
            "state2": self.model.encode("{").numpy().ravel(),
            "state3": self.model.encode(f'"prompt": {prompt_json},').numpy().ravel(),
            "state4": self.model.encode(f'"name": {name_json},').numpy().ravel(),
            "state5": self.model.encode(f'"parameters": {parameters_json}').numpy().ravel(),
            "state6": self.model.encode(f'{commas}').numpy().ravel(),
            "state7": self.model.encode("]").numpy().ravel()
        }
        if json_out.startswith("["):
            stats.pop("state1")
        if prompt < len(self.inpt_prompts) - 1:
            stats.pop("state7")

        for t_ids in stats.values():
            logits = self._get_logits(self.inpt_prompts[prompt].get("prompt"))
            for t_id in t_ids:
                try:
                    logits[t_id]
                    json_out += self.model.decode(t_id)
                except:
                    raise Exception
        return json_out


    def ano(self):
        with open(self.args.input) as f:
            self.inpt_prompts = json.load(f)
        with open(self.args.output, "w+") as f:
            json_out = ''
            prompt = 0
            while not json_out.endswith("]"):
                json_out = self.json_gen(json_out, prompt)
                prompt += 1
            data = json.loads(json_out)
            json.dump(data, f, indent=2)
