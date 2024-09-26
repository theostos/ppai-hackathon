import re
import json

from Levenshtein import distance
import yaml


content_filter = {}
with open('test.yaml', 'r', encoding='utf-8') as file:
    content = yaml.safe_load(file)
    for entry in content:
        content_filter[entry['context']] = entry
with open('test_s.yaml', 'r', encoding='utf-8') as file:
    content += yaml.safe_load(file)
    for entry in content:
        content_filter[entry['context']] = entry

content = list(content_filter.values())
miss = 0

def extract_subtext(text):
    # Find the largest substring that starts and ends with double quotes
    matches = re.findall(r'"(.*)"', text)
    if matches:
        # Return the longest match
        return max(matches, key=len)
    else:
        return None

result = []

for entry in content:
    context = entry['context']
    context_obf = entry['llm_output'][0]
    context_obf = extract_subtext(context_obf)
    if context_obf:
        context_obf = re.sub('<sensitive[^>]', '<sensitive>', context_obf)
        context_obf = re.sub('<\/sensitive[^>]', '<\/sensitive>', context_obf)

        context_obf_bis = context_obf.replace('<sensitive>', '')
        context_obf_bis = context_obf_bis.replace('</sensitive>', '')

        dist = distance(context, context_obf_bis)
        if dist < 30:
            entry['context_llm'] = context_obf
            del entry['llm_output']
            result.append(entry)
        else:
            miss += 1

with open(f"squad.json", "w", encoding='utf-8') as file:
    json.dump({"train": result}, file, indent=4)
print(miss)
print(len(content))