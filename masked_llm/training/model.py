import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..models.modeling_llamask import LlamaskForCausalLM
from ..models.tokenizer_utils import prepare_tokenizer
import yaml
from accelerate import Accelerator

def load_model() -> any:
    '''
    Load and return model using args from model_config.yaml in relative_path
    '''
    current_script_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_script_dir, 'config_model.yaml')
    with open(file_path, 'r', encoding='utf-8') as file:
        config_model = yaml.safe_load(file)

    model_name = config_model['model_name']
    
    peft_config = config_model['peft_config']
    peft_config = LoraConfig(**peft_config)

    model = LlamaskForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": Accelerator().process_index},
        torch_dtype=torch.bfloat16
    )

    # model.score.weight.requires_grad_(False)
    model.config.use_cache = False # silence the warnings. Please re-enable for inference!
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    return model

def load_tokenizer() -> any:
    '''
    Load and return tokenizer using args from model_config.yaml in relative_path
    '''
    current_script_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_script_dir, 'config_model.yaml')
    with open(file_path, 'r', encoding='utf-8') as file:
        config_model = yaml.safe_load(file)
    padding_side = config_model['padding_side']
    tokenizer_name = config_model['tokenizer_name']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, padding_side=padding_side, TOKENIZERS_PARALLELISM=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    prepare_tokenizer(tokenizer)
    return tokenizer

