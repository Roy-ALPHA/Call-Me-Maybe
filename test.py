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
    "fn_add_numbers",
    "fn_greet",
    "fn_reverse_string",
    "fn_get_square_root",
    "fn_substitute_string_with_regex"
]

test = {
    "prompt": prompt,
    "name": "", 
}

with open("data/input/functions_definition.json") as f, open("prompt") as p:
    funcs = json.load(f)
    prompt = "prompt: " + f"{prompt}" + "\nfrom this prompt, select the best matching dict from this list of dicts:\n" + funcs.__repr__()


top_scores = [np.sum(logits[model.encode(func["name"]).numpy().ravel()]) for func in funcs]

top_scores = np.array(top_scores)



for _ in range(3):
    top_score = np.argmax(top_scores)
    trie.insert(model.encode(functions[top_score]).numpy().ravel().tolist())
    top_scores[top_score] = float("-inf")
    # print(functions[top_score])

func_name = []
cur_node = trie.root
tmp = prompt
print(prompt)
# while not cur_node.is_end and cur_node.children:

#     logits = model.get_logits_from_input_ids(model.encode(prompt).numpy().ravel().tolist() + func_name)

#     allowed_tokens = trie.get_allowed_next_tokens(cur_node)

#     valid_token_scores = mask_invalid_tokens(logits, allowed_tokens)

#     best_token = max(valid_token_scores, key=valid_token_scores.get)

#     func_name.append(int(best_token))

#     # test["name"] += model.decode(int(best_token))

#     cur_node = trie.get_node(best_token, cur_node)


# print([model.decode(func) for func in func_name])

# for func in funcs:
#     if func["name"] == test["name"]:
#         test["paramters"] = func["parameters"]
# prompt = "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"
# prompt = "selected function:\n" + funcs[-1]["name"] + "\nfunction description:\n" + funcs[-1]["description"] + "\nParameters:\n" + funcs[-1]["parameters"].__repr__()  + "\nprompt:\n" + prompt + "\nYour task is to extract argument values for a selected function.\n" + "current parameter:\n" + '"source_string": '
# print(prompt)
# logits = model.get_logits_from_input_ids(model.encode(prompt).numpy().ravel().tolist())
# logits = np.array(logits)
cur = []
while True:
    logits = model.get_logits_from_input_ids(model.encode(prompt).numpy().ravel().tolist() + cur)
    logits = np.array(logits)

    best = np.argmax(logits)

    cur.append(best)
    # if model.decode(best).isdigit():
    print(model.decode(best))
    #     break

    # logits[best] = float("-inf")

    sleep(0.3)

# # print([model.decode(func) for func in func_name])