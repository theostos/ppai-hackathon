import os
import random

current_script_dir = os.path.dirname(__file__)
PROMPT_SQUAD_PATH = os.path.join(current_script_dir, 'prompt_squad.txt')
PROMPT_SLIMORCA_PATH = os.path.join(current_script_dir, 'prompt_slimorca.txt')

with open(PROMPT_SQUAD_PATH, 'r') as prompt_file:
    PROMPT_SQUAD = prompt_file.read()

with open(PROMPT_SLIMORCA_PATH, 'r') as prompt_file:
    PROMPT_SLIMORCA = prompt_file.read()

def formatting_prompts_func_alternate(prompt_squad, prompt_slimorca, example):
    output_texts = []

    for i in range(len(example['context'])):
        if example['benchmark'][i] == 'squad':
            context = example['context_llm'][i]
            question = example['question_llm'][i]
            answer = example['answer'][i]
            new_prompt = prompt_squad.format(context=context, question=question, answer=answer)
            output_texts.append(new_prompt)
        if example['benchmark'][i] == 'slimorca':
            question = example['question_llm'][i]
            answer = example['answer'][i]
            new_prompt = prompt_slimorca.format(question=question, answer=answer)
            output_texts.append(new_prompt)
    return output_texts

formatting_prompts = lambda x: formatting_prompts_func_alternate(PROMPT_SQUAD, PROMPT_SLIMORCA, x)
