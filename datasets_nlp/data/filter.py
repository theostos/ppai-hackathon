import re
import json

from Levenshtein import distance
import yaml


content_filter = {}
with open('slimorca.json', 'r', encoding='utf-8') as file:
    content = json.load(file)['train']
    for entry in content:
        content_filter[entry['question']] = entry

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
d = distance("Title: ART IN WORDS Review: this book was great,so powerful the words sting your eyes, a superlative book. A meticulous and astonishing vivid creation of one man's journey from a kind of hell into a life among whited skinned aristocratic men into a kind of pleading dream. A graceful and moving memoir. if you will. poignant recollection, lyrical and evocative. more than a record of unusual events, it shows how in a matter of survival, the courageous reffusal to abandon the fifth as it truely and always will be. the reader is left in awe of the bravery, endurance and solidarity of which humans are capable, as well as the brutality, evil and devisiveness they can inflict. Is the review positive or negative?\nThe answer to this question is:", "The answer to this question is: <sensitive>positive</sensitive>.")

# print(d)
# exit()
for entry in content[2:]:
    context = entry['question']
    context_obf = entry['question_llm']
    context_obf = context_obf
    if context_obf:
        context_obf = re.sub('<sensitive[^>]', '<sensitive>', context_obf)
        context_obf = re.sub('<\/sensitive[^>]', '<\/sensitive>', context_obf)

        context_obf_bis = context_obf.replace('<sensitive>', '')
        context_obf_bis = context_obf_bis.replace('</sensitive>', '')

        dist = distance(context, context_obf_bis)
        if dist < 30:
            entry['question_llm'] = context_obf
            result.append(entry)
        else:
            miss += 1

with open(f"slimorca.json", "w", encoding='utf-8') as file:
    json.dump({"train": result}, file, indent=4)
