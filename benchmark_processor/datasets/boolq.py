import random
from collections import defaultdict

from datasets import load_dataset
import torch

from .utils import load_prompt_template

def agglomerate_boolq_by_context(dataset, max_len=1500):
    """ Agglomerates the BoolQ dataset by 'context', grouping multiple questions per context. """
    context_grouped = defaultdict(list)
    
    for example in dataset:
        context = example["passage"]
        question = example["question"]
        answer = example['answer']  # Using the first answer as the primary one
        
        # Append the question and answer to the corresponding context

        if len(context) < max_len and len(question) < max_len:
            context_grouped[context].append({"question": question, "answer": answer})
    
    result = [{"context": context, "questions": questions} for context, questions in context_grouped.items()]
    return result

def generate_sigma_prompts_boolq(sample, k=1):
    template_prompt_context = load_prompt_template('context')
    template_prompt_question = load_prompt_template('question')

    new_prompts_dict = defaultdict(list)
    new_prompts_dict['context'] = sample['context']
    for entry in template_prompt_context:
        content = entry['content'].format(context=sample['context'])
        new_prompts_dict['context_llm'].append((entry['role'], content))

    k = min(k, len(sample['questions']))
    sample_questions = random.sample(sample['questions'], k=k)
    new_prompts_dict['num_sample'] = k

    for entry in template_prompt_question:
        for k, entry_questions in enumerate(sample_questions, start=1):
            question = entry_questions['question']
            content = entry['content'].format(context=question)
            new_prompts_dict[f'question_llm_{k}'].append((entry['role'], content))

    for k, entry_answer in enumerate(sample_questions, start=1):
        new_prompts_dict[f'answer_{k}'] = entry_answer['answer']
        new_prompts_dict[f'question_{k}'] = entry_answer['question']

    return new_prompts_dict

def generate_evaluation_prompt(sample):
    pass

def evaluate_answer(probs, sample):
    """ Given a probs output, compute the accuracy"""
    if sample['answer']:
        key_probs = 'yes'
    else:
        key_probs = 'no'
    return probs[key_probs]

def sample_boolq(max_len=1500):
    """ Generates prompts"""
    boolq_dataset = load_dataset("google/boolq", split="validation")
    boolq_dataset = boolq_dataset.shuffle()
    dataset_agglo = agglomerate_boolq_by_context(boolq_dataset, max_len=max_len)
    return dataset_agglo

def _test_sample():
    dataset = sample_boolq()
    dataset_flatten = [{"context": context, **val[0]} for context, val in dataset.items()]
    for _ in range(3):
        entry = random.choice(dataset_flatten)
        print('####')
        for key, value in generate_sigma_prompts_boolq(entry).items():
            print(key)
            for role, content in value:
                print(role)
                print(content)
        print()

if __name__ == '__main__':
    _test_sample()