from llm_sdk import Small_LLM_Model
import numpy as np
from time import sleep
import json

def mask_invalid_tokens(logits, allowed_tokens):
    return {
        token: logits[int(token)] for token in allowed_tokens 
    }

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, tokens):
        node = self.root
        
        for token in tokens:
            if str(token) not in node.children:
                node.children[str(token)] = TrieNode()

            node = node.children[str(token)]
        
        node.is_end = True
    
    def get_allowed_next_tokens(self, node):
        return list(node.children.keys())

    def get_node(self, prefix, cur_node):
        return cur_node.children[prefix]



model = Small_LLM_Model()
trie = Trie()
prompt = "What is the sum of 265 and 345?"
functions = [
    "fn_add_numbers",
    "fn_greet",
    "fn_reverse_string",
    "fn_get_square_root",
    "fn_substitute_string_with_regex"
]
for func in functions:
    trie.insert(model.encode(func).numpy().ravel().tolist())

func_name = []
cur_node = trie.root

while True:
    logits = model.get_logits_from_input_ids(model.encode(prompt).numpy().ravel().tolist())

    allowed_tokens = trie.get_allowed_next_tokens(cur_node)

    valid_token_scores = mask_invalid_tokens(logits, allowed_tokens)

    best_token = max(valid_token_scores, key=valid_token_scores.get)

    func_name.append(int(list(best_token.key())[0]))

    cur_node = trie.get_node(list(best_token.key())[0])