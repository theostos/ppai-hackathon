import json


content_filter = {}
with open('slimorca.json', 'r', encoding='utf-8') as file:
    content = json.load(file)['train']
    for entry in content:
        content_filter[entry['question']] = entry

content = list(content_filter.values())
miss = 0
result = []

for entry in content:
    question = entry['question']
    question_llm = entry['question_llm']
    answer = entry['answer']

    entry['answer'] = " ".join(answer.split()[:35])
    result.append(entry)



with open(f"slimorca_cut.json", "w", encoding='utf-8') as file:
    json.dump({"train": result}, file, indent=4)
