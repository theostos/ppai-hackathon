import os
from pathlib import Path
import json

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from masked_llm.models.tokenizer_utils import generate_custom_mask, prepare_tokenizer
from datasets_nlp.evaluation import generate_prompt, evaluate_boolq, evaluate_mmlu
from .models.modeling_llamask import LlamaskForCausalLM
from .models.tokenizer_utils import prepare_tokenizer
from .training.trainer import load_trainer

PARENT_DIR = Path(__file__).parent.parent

with open(os.path.join(PARENT_DIR, 'datasets_nlp/data/mmlu.json'), 'r') as file:
    dt_mmlu = json.load(file)['train']
with open(os.path.join(PARENT_DIR, 'datasets_nlp/data/boolq.json'), 'r') as file:
    dt_boolq = json.load(file)['train']

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
EXP_PATH = "results"
device = 'cuda'
trainer, model, model_dir = load_trainer(EXP_PATH, 'logs')
trainer.train()
