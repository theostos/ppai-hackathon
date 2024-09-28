import logging
import time

import torch
from load_huggingface import get_gpt2_tokenizer
from qgpt2_models import SingleHeadQGPT2ModelSimulationHybrid

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# gpt2_model = get_gpt2_model("gpt2_model")
gpt2_tokenizer = get_gpt2_tokenizer("gpt2_tokenizer")

proj_single_head_qgpt2 = SingleHeadQGPT2ModelSimulationHybrid.from_pretrained(
    "gpt2_model", n_bits=7, use_cache=False, num_fhe=1
)

proj_single_head_qgpt2.set_fhe_mode(fhe="disable")

input_sentence = "Computations on encrypted data can help"

input_token_indexes = gpt2_tokenizer.encode(input_sentence)
input_ids = torch.tensor(input_token_indexes).unsqueeze(0)
input_ids = input_ids[:,:2]




# output_ids_clear = proj_single_head_qgpt2.generate(input_ids, max_new_tokens=4)
# gpt2_tokenizer.decode(output_ids_clear[0])

# output_logits_clear = proj_single_head_qgpt2(input_ids).logits
circuit_single_head = proj_single_head_qgpt2.compile(input_ids, msbs_round=6)
print("execute")
proj_single_head_qgpt2.set_fhe_mode(fhe="execute")

start = time.time()
output_logits_fhe = proj_single_head_qgpt2(input_ids).logits
end = time.time()
execution_time = end - start

print(f"Time taken to execute the single head in FHE: {execution_time:.2f} seconds")
