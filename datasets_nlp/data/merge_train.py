import json
import random

content = []
filenames = [('slimorca', 'slimorca_cut.json'), ('squad', 'squad.json')]
for kind, filename in filenames:
    with open(filename, 'r', encoding='utf-8') as file:
        content_bis = json.load(file)['train']
        for entry in content_bis:
            entry['benchmark'] = kind
            content.append(entry)

            if kind == 'slimorca':
                entry['context_llm'] = ''
                entry['context'] = ''
                
            if kind == 'squad':
                if 'Q1' in entry:
                    entry['question_llm'] = '<sensitive>' + entry['Q1'] + '</sensitive>'
                    entry['answer'] = entry['A1']
                    entry['question'] = ''
                    del entry['Q1']
                    del entry['A1']
                    del entry['Q2']
                    del entry['A2']
                if 'answer_1' in entry:
                    entry['question'] = ''
                    entry['answer'] = entry['answer_1']
                    entry['question_llm'] = entry['question_llm_1']


                    del entry['question_llm_1']
                    del entry['answer_1']
                    if 'answer_2' in entry:
                        del entry['question_llm_2']
                        del entry['answer_2']
random.shuffle(content)
print(len(content))
with open(f"merge_slimorca_squad_bis.json", "w", encoding='utf-8') as file:
    json.dump({"train": content}, file, indent=4)
