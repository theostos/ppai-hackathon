import random

from collections import defaultdict
from datasets import load_dataset

from .utils import load_prompt_template

def filter_mmlu(dataset, max_len=1500):
    """ Filter the mmlu dataset"""
    dataset_filtered = []
    for example in dataset:
        question = example['question']
        entry = {'question':question}
        idx_incorrect = 1
        for k, choice in enumerate(example['choices']):
            if k == example['answer']:
                entry['correct_answer'] = choice
            else:
                entry[f'incorrect_answer_{idx_incorrect}'] = choice
                idx_incorrect += 1
        
        if len(question) < max_len:
            dataset_filtered.append(entry)
    return dataset_filtered

def generate_sigma_prompts_mmlu(sample):
    template_prompt_context = load_prompt_template('context')

    new_prompts_dict = defaultdict(list)
    for entry in template_prompt_context:
        content = entry['content'].format(context=sample['question'])
        new_prompts_dict['question_llm'].append((entry['role'], content))

    new_prompts_dict.update(sample)
    return new_prompts_dict

def sample_mmlu(max_len=1500):
    """ Generates prompts"""
    mmlu_dataset = load_dataset("cais/mmlu", 'all', split='validation')
    mmlu_dataset = mmlu_dataset.shuffle()
    dataset_filtered = filter_mmlu(mmlu_dataset, max_len=max_len)
    return dataset_filtered

def _test_sample():
    dataset = sample_mmlu()
    for _ in range(3):
        entry = random.choice(dataset)
        print('####')
        for key, value in generate_sigma_prompts_mmlu(entry).items():
            print(key)
            for role, content in value:
                print(role)
                print(content)
        print()

if __name__ == '__main__':
    _test_sample()