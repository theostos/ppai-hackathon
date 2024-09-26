import random

from collections import defaultdict
from datasets import load_dataset

from .utils import load_prompt_template

def filter_gsm8k(dataset, max_len=1500):
    """ Agglomerates the GSM8k dataset by 'context', grouping multiple questions per context. """
    dataset_filtered = []
    
    for example in dataset:
        question = example["question"]
        answer, final_answer = example['answer'].split('#### ')
        # Append the question and answer to the corresponding context
        if len(question) < max_len:
            dataset_filtered.append({"question": question, "answer": answer, "final_answer": final_answer})
    return dataset_filtered

def generate_sigma_prompts_gsm8k(sample):
    template_prompt_question = load_prompt_template('context')

    new_prompts_dict = defaultdict(list)
    for entry in template_prompt_question:
        content = entry['content'].format(context=sample['question'])
        new_prompts_dict['question_llm'].append((entry['role'], content))
    new_prompts_dict.update(sample)

    return new_prompts_dict

def sample_gsm8k(max_len=1500):
    """ Generates prompts """
    gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_dataset = gsm8k_dataset.shuffle()
    dataset_filtered = filter_gsm8k(gsm8k_dataset, max_len=max_len)
    return dataset_filtered

def _test_sample():
    dataset_filtered = sample_gsm8k()
    for _ in range(3):
        entry = random.choice(dataset_filtered)
        print('####')
        for key, value in generate_sigma_prompts_gsm8k(entry).items():
            print(key)
            for role, content in value:
                print(role)
                print(content)
        print()

if __name__ == '__main__':
    _test_sample()