import random

from collections import defaultdict
from datasets import load_dataset

from .utils import load_prompt_template

def filter_gpqa(dataset, max_len=1500):
    """ Filter the gpqa dataset"""
    dataset_filtered = []
    wanted_keys = [
        ('Question', 'question'),
        ('Correct Answer', 'correct_answer'),
        ('Incorrect Answer 1', 'incorrect_answer_1'),
        ('Incorrect Answer 2', 'incorrect_answer_2'),
        ('Incorrect Answer 3', 'incorrect_answer_3')
        ]

    for example in dataset:
        question = example['Question']
        if len(question) < max_len:
            entry = {key_2:example[key_1] for key_1, key_2 in wanted_keys}
            dataset_filtered.append(entry)
    return dataset_filtered

def generate_sigma_prompts_gpqa(sample):
    template_prompt_context = load_prompt_template('context')
    new_prompts_dict = defaultdict(list)
    for entry in template_prompt_context:
        content = entry['content'].format(context=sample['question'])
        new_prompts_dict['question_llm'].append((entry['role'], content))
    new_prompts_dict.update(sample)
    return new_prompts_dict

def sample_gpqa(max_len=1500):
    """ Generates prompts"""
    gpqa_dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    gpqa_dataset = gpqa_dataset.shuffle()
    dataset_filtered = filter_gpqa(gpqa_dataset, max_len=max_len)
    return dataset_filtered

def _test_sample():
    dataset = sample_gpqa()
    for _ in range(3):
        entry = random.choice(dataset)
        print('####')
        for key, value in generate_sigma_prompts_gpqa(entry).items():
            print(key)
            for role, content in value:
                print(role)
                print(content)
        print()

if __name__ == '__main__':
    _test_sample()