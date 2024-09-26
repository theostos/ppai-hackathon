from functools import partial
import os

import yaml
from datasets import load_dataset as load_dataset_fun_hf

from .model import load_tokenizer, load_model
from .prompt import formatting_prompts

def load_dataset():
    current_script_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_script_dir, 'config_dataset.yaml')
    with open(file_path, 'r', encoding='utf-8') as file:
        config_dataset = yaml.safe_load(file)
    dataset_path = config_dataset['dataset_path']
    dataset = load_dataset_fun_hf('json', data_files=dataset_path, field="train")["train"]
    return dataset
