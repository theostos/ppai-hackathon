# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama import LlamaForCausalLM

from .tokenizers import CustomLlamaTokenizer

class LlamaskForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.special_tokens = nn.Embedding(2, config.hidden_size) # 0 -> mask encoding, 1 -> buffer token
        self.post_init()

    def generate(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: torch.Tensor = None,
            position_ids: torch.LongTensor = None,
            max_tokens: int=32,
            temperature: float=1.0,
            **kwargs
    ):
        eos_token_ids = torch.tensor(self.config.eos_token_id, device=input_ids.device)
        for _ in range(max_tokens):
            prev_tokens = input_ids[:,-1]
            is_eos = torch.isin(prev_tokens, eos_token_ids)
            if is_eos.all():
                break
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            logits = outputs['logits'][:,-1,:]/temperature

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).int()

            batch_size, seq_len, _ = attention_mask.shape
            expanded_mask = torch.zeros(batch_size, seq_len + 1, seq_len + 1, dtype=attention_mask.dtype, device=attention_mask.device)

            # Step 1: Copy the existing attention mask (top-left block of the expanded mask)
            expanded_mask[:, :seq_len, :seq_len] = attention_mask

            # Step 2: Copy the last row of the original attention mask into the new row (excluding the last position)
            expanded_mask[:, seq_len, :seq_len] = attention_mask[:, -1, :]

            # Step 3: Set the diagonal of the new token to attend to all previous tokens by setting the new last element to 1
            expanded_mask[:, seq_len, seq_len] = 1
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            input_ids[is_eos, -1] = eos_token_ids[0]
            expanded_mask[is_eos, -1, :] = 0

            last_column_position_ids = position_ids[:, -1]
            new_position_ids = last_column_position_ids + 1
            new_position_ids = new_position_ids.unsqueeze(1)

            # Concatenate the new column to the original position_ids tensor along the sequence dimension (dim=1)
            position_ids = torch.cat((position_ids, new_position_ids), dim=1)

            attention_mask = expanded_mask
        return input_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        num_buffer_token: Optional[int] = 0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        batch_size = input_ids.shape[0]

        # print("BEWARE PRIVACY TAG DISABLE")
        # privacy_tag = self.special_tokens(torch.tensor([0], device=input_ids.device))
        # buffer_token = self.special_tokens(torch.tensor([0], device=input_ids.device)).unsqueeze(0)
        
        inputs_embeds = self.model.embed_tokens(input_ids)
        # buffer_tokens = buffer_token.repeat(batch_size, num_buffer_token, 1)
        # inputs_embeds = torch.cat([inputs_embeds, buffer_tokens], dim=1)
        
        # inputs_embeds[attention_mask[:,-1,:]==0] = inputs_embeds[attention_mask[:,-1,:]==0] + privacy_tag

        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.to(inputs_embeds.dtype)
        attention_mask = attention_mask.masked_fill(attention_mask == 0, -1e9)
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
        
        outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
        return outputs

if __name__ == '__main__':
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device = 'cuda:2'

    model = LlamaskForCausalLM.from_pretrained(model_id, torch_dtype= torch.bfloat16)
    model = model.to(device)
    tokenizer = CustomLlamaTokenizer.from_pretrained(model_id)
    tokenizer.add_privacy_tokens()

    # Tokenize some text with the custom tokenizer
    prompt = """<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
What is the <privacy>capital</privacy> of <privacy>Tonga</privacy> ?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    prompt_clear = prompt.replace('<privacy>', '')
    prompt_clear = prompt_clear.replace('</privacy>', '')

    model_inputs = tokenizer([prompt_clear, prompt]).to(device)

    outputs = model.generate(temperature=0.7, max_tokens=64, **model_inputs)
    outputs = outputs[:, model_inputs['input_ids'].shape[1]:]
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for answer in result:
        print(answer)