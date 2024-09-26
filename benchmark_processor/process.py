from itertools import zip_longest

import yaml
from tqdm import tqdm

from .client.client import InferenceLLM


from .datasets.boolq import sample_boolq, generate_sigma_prompts_boolq
from .datasets.gpqa import sample_gpqa, generate_sigma_prompts_gpqa
from .datasets.gsm8k import sample_gsm8k, generate_sigma_prompts_gsm8k
from .datasets.mmlu import sample_mmlu, generate_sigma_prompts_mmlu
from .datasets.slimorca import sample_slimorca, generate_sigma_prompts_slimorca
from .datasets.squad import sample_squad, generate_sigma_prompts_squad

VLLM_SERVER_URL = "http://127.0.0.1:5000/inference"

llm = InferenceLLM(VLLM_SERVER_URL)

if __name__ == "__main__":
    # samples_boolq = sample_boolq()
    # samples_gpqa = sample_gpqa()
    # samples_gsm8k = sample_gsm8k()
    # samples_mmlu = sample_mmlu()
    samples_slimorca = sample_slimorca()
    samples_squad = sample_squad()

    samples = zip(samples_slimorca, samples_squad) #, samples_slimorca, samples_squad)

    for k, (s_slimorca, s_squad) in tqdm(enumerate(samples)):
        res = []
        try:
            # prompts_boolq = generate_sigma_prompts_boolq(s_boolq)
            # prompts_gpqa = generate_sigma_prompts_gpqa(s_gpqa)
            # prompts_gsm8k = generate_sigma_prompts_gsm8k(s_gsm8k)
            # prompts_mmlu = generate_sigma_prompts_mmlu(s_mmlu)
            prompts_slimorca = generate_sigma_prompts_slimorca(s_slimorca)
            prompts_squad = generate_sigma_prompts_squad(s_squad)

            prompts = [
                # (s_boolq, prompts_boolq, 'BoolQ'),
                # (s_gpqa, prompts_gpqa, 'GPQA'),
                # (s_gsm8k, prompts_gsm8k, 'GSM8K'), 
                # (s_mmlu, prompts_mmlu, 'MMLU'),
                (s_slimorca, prompts_slimorca, 'SlimOrca'), 
                # (s_squad, prompts_squad, 'SQuAD')
            ]
            
            prompts = [prompt_triplet for prompt_triplet in prompts if prompt_triplet[0]]

            for (sample, prompts_dict, bench_name) in prompts:
                print(bench_name)
                for key, prompt in prompts_dict.items():
                    if 'llm' in key:
                        sample[key] = llm([prompt])[0]
                sample["benchmark"] = bench_name
                res.append(sample)

            with open(f"output/output_{k}.yaml", "w", encoding='utf-8') as file:
                yaml.dump(res, file, sort_keys=False)
        except Exception as e:
            with open(f"output/output_{k}.yaml", "w", encoding='utf-8') as file:
                yaml.dump(res, file, sort_keys=False)
            print(e)
