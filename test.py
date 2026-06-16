from llm_sdk import Small_LLM_Model
import numpy as np
from time import sleep
import json

model = Small_LLM_Model()
text = "remove user 123"
func1 = model.encode("delete_user").numpy().ravel()
func2 = model.encode("create_user").numpy().ravel()
func3 = model.encode("search_user").numpy().ravel()
func4 = model.encode("remove_user").numpy().ravel()
nm = model.encode(text).numpy().ravel()
l = model.get_logits_from_input_ids(nm.tolist())
with open(model.get_path_to_vocab_file()) as f:
    vocab = json.load(f)

l = np.array(l)
# while True:
print("func1" ,np.sum(l[func1]))
print("func2" ,np.sum(l[func2]))
print("func3" ,np.sum(l[func3]))
print("func4" ,np.sum(l[func4]))

