import os
import yaml
PREPROCESS_ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(PREPROCESS_ROOT, 'prompts')

def load_prompt_template(name):
    prompt_path = os.path.join(PROMPT_PATH, f'template_{name}.yaml')
    with open(prompt_path, mode='r', encoding='utf-8') as file:
        return yaml.safe_load(file)