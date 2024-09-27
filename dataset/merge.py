import json

content_filter = {}
filenames = ['slimorca_1.json', 'slimorca_2.json', 'slimorca_3.json']
for filename in filenames:
    with open(filename, 'r', encoding='utf-8') as file:
        content = json.load(file)['train']
        print(len(content))
        if 'Q1' in content:
            content['question_1'] = content['Q1']
            content['answer_1'] = content['A1']
            del content['Q1']
            del content['A1']
        if 'Q2' in content:
            content['question_2'] = content['Q2']
            content['answer_2'] = content['A2']
            del content['Q2']
            del content['A2']
        for entry in content:
            content_filter[entry['question']] = entry

content = list(content_filter.values())
print(len(content))
with open(f"slimorca.json", "w", encoding='utf-8') as file:
    json.dump({"train": content}, file, indent=4)
