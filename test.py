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
prompt = "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"
logits = np.array(model.get_logits_from_input_ids(model.encode(prompt).numpy().ravel().tolist()))

functions = [
    "fn_add_numbers Add two numbers together and return their sum.",
    "fn_greet Generate a greeting message for a person by name.",
    "fn_reverse_string Reverse a string and return the reversed result.",
    "fn_get_square_root Calculate the square root of a number.",
    "fn_substitute_string_with_regex Replace all occurrences matching a regex pattern in a string."
]
# for func in functions:
#     trie.insert(model.encode(func).numpy().ravel().tolist())
# top_scores = []

# for func in functions:
#     tokens = model.encode(func).numpy().ravel()
#     score = np.sum(logits[tokens])
#     top_scores.append(score / len(tokens))


top_scores = [np.sum(logits[model.encode(func).numpy().ravel()]) for func in functions]

top_scores = np.array(top_scores)

text = "functions: "

for _ in range(3):
    top_score = np.argmax(top_scores)
    trie.insert(model.encode(functions[top_score]).numpy().ravel().tolist())
    top_scores[top_score] = float("-inf")
    print(functions[top_score])
    text += functions[top_score]

func_name = []
cur_node = trie.root

while not cur_node.is_end and cur_node.children:

    logits = model.get_logits_from_input_ids(model.encode(prompt).numpy().ravel().tolist())

    allowed_tokens = trie.get_allowed_next_tokens(cur_node)

    valid_token_scores = mask_invalid_tokens(logits, allowed_tokens)
    # print([(model.decode(int(token)), logits[int(token)]) for token in valid_token_scores.keys()])
    best_token = max(valid_token_scores, key=valid_token_scores.get)

    func_name.append(int(best_token))

    cur_node = trie.get_node(best_token, cur_node)


print([model.decode(func) for func in func_name])