import os
import random

current_script_dir = os.path.dirname(__file__)
PROMPT_PATH = os.path.join(current_script_dir, 'prompt_squad.txt')

with open(PROMPT_PATH, 'r') as prompt_file:
    PROMPT_TXT = prompt_file.read()

def formatting_prompts_func_alternate(prompt, example):
    output_texts = []
    for i in range(len(example['context'])):
        context = example['context'][i]
        question = example['Q1'][i]
        answer = example['A1'][i]
        new_prompt = prompt.format(context=context, question=question, answer=answer)
        output_texts.append(new_prompt)
    return output_texts

formatting_prompts = lambda x: formatting_prompts_func_alternate(PROMPT_TXT, x)
