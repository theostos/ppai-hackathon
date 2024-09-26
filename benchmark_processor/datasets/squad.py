from collections import defaultdict
import random

from datasets import load_dataset
from .utils import load_prompt_template

def agglomerate_squad_by_context(dataset, max_len=1500):
    """ Agglomerates the SQuAD dataset by 'context', grouping multiple questions per context. """
    context_grouped = defaultdict(list)
    
    for example in dataset:
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]  # Using the first answer as the primary one
        
        # Append the question and answer to the corresponding context
        if len(context) < max_len and len(question) < max_len:
            context_grouped[context].append({"question": question, "answer": answer})
    
    result = [{"context": context, "questions": questions} for context, questions in context_grouped.items()]
    return result

def generate_sigma_prompts_squad(sample, k=2):
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

def sample_squad(max_len=1500):
    """ Generates and saves the few-shot prompts to a text file. """
    squad_dataset = load_dataset("squad", split="validation")
    squad_dataset = squad_dataset.shuffle()
    dataset_agglo = agglomerate_squad_by_context(squad_dataset, max_len=max_len)
    return dataset_agglo

def _test_sample():
    dataset_agglo = sample_squad()
    for _ in range(3):
        entry = random.choice(dataset_agglo)
        print('####')
        for key, value in generate_sigma_prompts_squad(entry).items():
            print(key)
            if isinstance(value, list):
                for role, content in value:
                    print(role)
                    print(content)
            else:
                print(value)
        print()

if __name__ == '__main__':
    _test_sample()