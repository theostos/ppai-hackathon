import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from .models.modeling_llamask import LlamaskForCausalLM
from masked_llm.models.tokenizer_utils import generate_custom_mask, prepare_tokenizer
from datasets_nlp.evaluation import generate_prompt, evaluate_boolq, evaluate_mmlu

PARENT_DIR = Path(__file__).parent.parent

with open(os.path.join(PARENT_DIR, 'datasets_nlp/data/mmlu.json'), 'r') as file:
    dt_mmlu = json.load(file)['train']
with open(os.path.join(PARENT_DIR, 'datasets_nlp/data/boolq.json'), 'r') as file:
    dt_boolq = json.load(file)['train']

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
prepare_tokenizer(tokenizer)

model = LlamaskForCausalLM.from_pretrained(model_id, torch_dtype= torch.bfloat16)
model = model.to(device)
# model.load_adapter('results/checkpoint-150')


with torch.no_grad():
    score_baseline = 0.
    score_masked = 0.
    for entry in tqdm(dt_mmlu):
        prompt = generate_prompt('mmlu', entry)

        prompt_clear = prompt.replace('<sensitive>', '')
        prompt_clear = prompt_clear.replace('</sensitive>', '')
        model_inputs = generate_custom_mask(tokenizer, [prompt, prompt_clear], device, padding_side='left')
        logits = model(**model_inputs, num_buffer_token=0)['logits']
        probs = torch.softmax(logits[:,-1,:]/1e-5, dim=-1) # ~= max
        score = evaluate_mmlu(entry, probs, tokenizer)
        score_masked += score[0]
        score_baseline += score[1]
    
    print(f"Score MMLU baseline: {score_baseline/len(dt_mmlu)}")
    print(f"Score MMLU no FT masked: {score_masked/len(dt_mmlu)}")

    score_baseline = 0.
    score_masked = 0.
    for entry in tqdm(dt_boolq):
        prompt = generate_prompt('boolq', entry)
        prompt_clear = prompt.replace('<sensitive>', '')
        prompt_clear = prompt_clear.replace('</sensitive>', '')

        model_inputs = generate_custom_mask(tokenizer, [prompt, prompt_clear], device, padding_side='left')
        logits = model(**model_inputs, num_buffer_token=0)['logits']
        probs = torch.softmax(logits[:,-1,:]/1e-5, dim=-1) # ~= max

        score = evaluate_boolq(entry, probs, tokenizer)
        score_masked += score[0]
        score_baseline += score[1]
    
    print(f"Score BoolQ baseline: {score_baseline/len(dt_mmlu)}")
    print(f"Score BoolQ no FT masked: {score_masked/len(dt_mmlu)}")