
import os
from collections import defaultdict
import string
import random

import yaml
PREPROCESS_ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(PREPROCESS_ROOT, 'prompts')

def load_benchmark_template(name):
    prompt_path = os.path.join(PROMPT_PATH, f'prompt_{name}.yaml')
    with open(prompt_path, mode='r', encoding='utf-8') as file:
        return yaml.safe_load(file)['content']

def safe_format(s, entry):
    # Create a Formatter object
    formatter = string.Formatter()
    
    # Extract the field names from the string
    keys_in_string = {field_name for _, field_name, _, _ in formatter.parse(s) if field_name}
    
    # Filter the dictionary to only include the relevant keys
    filtered_entry = {k: v for k, v in entry.items() if k in keys_in_string}
    
    # Use str.format with the filtered dictionary
    return s.format(**filtered_entry)

def _filter_entry_mmlu(entry):
    answers = [(False, entry['incorrect_answer_1']), (False, entry['incorrect_answer_2']), (False, entry['incorrect_answer_3']), (True, entry['correct_answer'])]
    random.shuffle(answers)
    entry['1'] = answers[0][0]
    entry['2'] = answers[1][0]
    entry['3'] = answers[2][0]
    entry['4'] = answers[3][0]

    entry['answer_1'] = '<sensitive>' + answers[0][1] + '</sensitive>'
    entry['answer_2'] = '<sensitive>' + answers[1][1] + '</sensitive>'
    entry['answer_3'] = '<sensitive>' + answers[2][1] + '</sensitive>'
    entry['answer_4'] = '<sensitive>' + answers[3][1] + '</sensitive>'

    entry['question'] = entry['question_llm']

def _filter_entry_boolq(entry):
    entry['question'] = entry['question_llm']

def generate_prompt(benchmark, entry):
    template_prompt = load_benchmark_template(benchmark)
    if benchmark == 'mmlu':
        _filter_entry_mmlu(entry)
    if benchmark == 'boolq':
        _filter_entry_boolq(entry)
    return safe_format(template_prompt, entry)

def evaluate_mmlu(entry, probs, tokenizer):
    a_token = tokenizer.encode(' 1')[-1]
    b_token = tokenizer.encode(' 2')[-1]
    c_token = tokenizer.encode(' 3')[-1]
    d_token = tokenizer.encode(' 4')[-1]

    score = probs[:, a_token]*entry['1'] + probs[:, b_token]*entry['2'] + probs[:, c_token]*entry['3'] + probs[:, d_token]*entry['4']
    return score

def evaluate_boolq(entry, probs, tokenizer):
    yes_token = tokenizer.encode(' 1')[-1]
    no_token = tokenizer.encode(' 0')[-1]

    score = probs[:, yes_token]*entry['answer'] + probs[:, no_token]*(not entry['answer'])
    return score