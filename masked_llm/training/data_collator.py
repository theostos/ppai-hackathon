import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from transformers import DataCollatorForLanguageModeling

from ..models.tokenizer_utils import generate_custom_mask_input_ids

class DataCollatorForCompletionOnlyLMask(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = {}
        input_ids_list = [entry['input_ids'] for entry in examples]
        input_models, full_attn_mask = generate_custom_mask_input_ids(self.tokenizer, input_ids_list)
        batch["input_ids"], batch["attention_mask"] = input_models['input_ids'], input_models['attention_mask']
        batch["position_ids"] = (full_attn_mask.cumsum(1) - 1)
        batch['labels'] = batch["input_ids"].clone()
        batch['labels'][full_attn_mask==0] = self.ignore_index
        
        for i in range(len(examples)):
            response_token_ids_start_idx = None

            for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                if (
                    self.response_token_ids
                    == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                ):
                    response_token_ids_start_idx = idx
            if response_token_ids_start_idx is None:
                warnings.warn(
                    f"Could not find response key `{self.response_template}` in the "
                    f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                batch["labels"][i, :] = self.ignore_index
            else:
                response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                # Make pytorch loss function ignore all tokens up through the end of the response key
                batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index
            
            seq_len = batch['position_ids'][i, -1].int()
            batch['attention_mask'][i, response_token_ids_end_idx:seq_len, :response_token_ids_end_idx] = 1
            batch['attention_mask'][i, response_token_ids_end_idx:seq_len, response_token_ids_end_idx:seq_len] = torch.tril(torch.ones(seq_len-response_token_ids_end_idx, seq_len-response_token_ids_end_idx))
        # print(batch['input_ids'][0,:])
        # print(batch['labels'][0,:])
        # torch.set_printoptions(threshold=10_000)
        # print(batch['attention_mask'][0,:])
        # print()
        # exit()
        return batch
