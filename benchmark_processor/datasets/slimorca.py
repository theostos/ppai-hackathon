import random

from collections import defaultdict
from datasets import load_dataset

from .utils import load_prompt_template

def filtered_slimorca(dataset, max_len=1500):
    """ Filter Slimorca dataset"""
    dataset_filtered = []
    
    for example in dataset:
        conversations = example['conversations']
        
        human_entry = ''
        assistant_entry = ''
        for conversation_entry in conversations:
            if conversation_entry['from'] == 'human':
                human_entry = conversation_entry['value']
            elif conversation_entry['from'] == 'gpt':
                assistant_entry = conversation_entry['value']
        
        if human_entry and assistant_entry and len(human_entry) < max_len:
            dataset_entry = {'question': human_entry, 'answer': assistant_entry}
            dataset_filtered.append(dataset_entry)
    return dataset_filtered

def generate_sigma_prompts_slimorca(sample):
    template_prompt_context = load_prompt_template('context')

    new_prompts_dict = defaultdict(list)
    for entry in template_prompt_context:
        content = entry['content'].format(context=sample['question'])
        new_prompts_dict['question_llm'].append((entry['role'], content))
    new_prompts_dict.update(sample)
    return new_prompts_dict

def sample_slimorca(max_len=1500):
    """ Generates and saves the few-shot prompts to a text file. """
    slimorca_dataset = load_dataset("Open-Orca/SlimOrca", split="train")
    slimorca_dataset = slimorca_dataset.shuffle()
    dataset_agglo = filtered_slimorca(slimorca_dataset, max_len=max_len)
    return dataset_agglo

def _test_sample():
    dataset = sample_slimorca()
    for _ in range(3):
        entry = random.choice(dataset)
        print('####')
        for key, value in generate_sigma_prompts_slimorca(entry).items():
            print(key)
            for role, content in value:
                print(role)
                print(content)
        print()

if __name__ == '__main__':
    _test_sample()