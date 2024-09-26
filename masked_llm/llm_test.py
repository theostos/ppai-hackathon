import torch

from transformers import AutoTokenizer
from .models.modeling_llamask import LlamaskForCausalLM
from masked_llm.models.tokenizer_utils import generate_custom_mask, prepare_tokenizer
from tqdm import tqdm

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = 'cuda:2'

model = LlamaskForCausalLM.from_pretrained(model_id, torch_dtype= torch.bfloat16)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

prepare_tokenizer(tokenizer)

prompt_2 = """<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
What is the <sensitive>capital</sensitive> of <sensitive>Tonga</sensitive> ?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

BATCH_SIZE = 2
NUM_SAMPLE = 40

prompts = [prompt_2]*BATCH_SIZE
model_inputs = generate_custom_mask(tokenizer, prompts, device)
score = 0

print(model(**model_inputs, num_buffer_token=2))
exit()
for _ in tqdm(range(NUM_SAMPLE)):
    
    outputs = model.generate(temperature=0.7, max_tokens=64, **model_inputs)
    outputs = outputs[:, model_inputs['input_ids'].shape[1]:]
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for answer in result:
        if 'Nuku' in answer:
            score += 1
print(score/(NUM_SAMPLE*BATCH_SIZE) * 100)