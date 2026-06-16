from llm_sdk import Small_LLM_Model
import numpy as np
from time import sleep
import json

model = Small_LLM_Model()
nm = model.encode("Greet shrek").numpy().ravel()
l = model.get_logits_from_input_ids(nm.tolist())
with open(model.get_path_to_vocab_file()) as f:
    vocab = json.load(f)

l = np.array(l)
while True:
    a = np.argmax(l)
    word = model.decode(a) 
    print(word)
    l[a] = float("-inf")
    sleep(1)
