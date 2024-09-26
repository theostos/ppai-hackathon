from typing import Any, List
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams
import yaml

class BaseLLM:
    """
    Simple LLM class
    """
    def __init__(self, config_path) -> None:
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        model = hf_hub_download(**self.config['llm_id'])
        self.llm = LLM(model=model, **self.config['llm_engine'])
        self.prompt_template = self.config['prompt_template']

    def __call__(self, prompts=None, temperature=0.7, max_tokens=128, append_helpful=True) -> List[str]:
        if not prompts:
            return []
        if append_helpful:
            prompts = [[('system', "You are a helpful assistant")] + prompt_list for prompt_list in prompts]
        
        prompts_raw = []
        for prompt_list in prompts:
            prompt_tot = ""
            for role, content in prompt_list:
                prompt_tot += self.prompt_template[role]['begin'] + content + self.prompt_template[role]['end']
            prompt_tot += self.prompt_template['assistant']['begin']
            prompts_raw.append(prompt_tot)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.generate(prompts_raw, sampling_params)

        result = [output.outputs[0].text for output in outputs]
        return result
