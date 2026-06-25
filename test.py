from llm_sdk import Small_LLM_Model
import numpy as np

with open("prompt_test.txt") as f:
    prompt = f.read()
model = Small_LLM_Model()

generated = []
while True:

    tokens = model.encode(prompt).numpy().ravel().tolist()

    logits = np.array(model.get_logits_from_input_ids(tokens + generated))

    best = np.argmax(logits)

    generated.append(best)

    logits[best] = float("-inf")

    print(model.decode(best), best)

