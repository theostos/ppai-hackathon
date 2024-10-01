from typing import List, Optional

from transformers import LlamaTokenizerFast
import torch

class CustomLlamaTokenizer(LlamaTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token = self.eos_token
        self.special_tokens = []
        self.privacy_beg_id = None
        self.privacy_end_id = None

    def add_privacy_tokens(self, token_beg="<privacy>", token_end="</privacy>"):
        """
        Add privacy tokens to the tokenizer.
        Store tokens ids.
        """

        assert not self.privacy_beg_id and not self.privacy_end_id, "Custom tokenizer class does not support update of privacy tokens for the moment"
        # Add the special tokens to the base tokenizer
        self.add_tokens([token_beg, token_end])
        # Update special tokens list in the custom tokenizer
        self.add_special_tokens({"additional_special_tokens": [token_beg, token_end]})

        encode_beg = self.convert_tokens_to_ids(token_beg)
        encode_end = self.convert_tokens_to_ids(token_end)

        self.privacy_beg_id = encode_beg
        self.privacy_end_id = encode_end

    def __call__(self, *args, split_idx_list: Optional[List[int]]=None, **kwargs):
        """"
        Override the call method to add custom mask generation.
        
        Args:
            args: Arguments for the tokenizer's base call.
            split_idx_list: List of index from which every tokens should be considered masked, by default None (last one only)
            kwargs: Keyword arguments for the tokenizer's base call.
        """
        # Call the base tokenizer to tokenize the inputs

        kwargs['padding'] = 'do_not_pad'
        encoding = super().__call__(*args, **kwargs)
        inputs = encoding["input_ids"]

        new_inputs = self.compute_custom_attention_mask(inputs, split_idx_list)

        encoding['input_ids'] = new_inputs['input_ids']
        encoding['attention_mask'] = new_inputs['attention_mask']
        encoding['full_attention_mask'] = new_inputs['full_attention_mask']
        encoding['position_ids'] = new_inputs['position_ids']

        return encoding

    def compute_custom_attention_mask(self, input_ids: torch.LongTensor, split_idx_list: Optional[List[int]]):
        """
        Given a prepared tokenizer (i.e. with privacy special tokens), a list of prompts with privacy special tokens,
        tokenize and generate custom masks for a privacy-compatible transformer.
        """
        new_input_ids, new_attention_masks, seq_len_list = [], [], []
        max_len = 0
        batch_size = len(input_ids)
        split_idx_list = split_idx_list or [-1]*batch_size

        for input_id, split_idx in zip(input_ids, split_idx_list):
            trigger_privacy = False
            new_input_id = []
            mask_pos_list = []
            idx = 0
            for token_id in input_id:
                if token_id == self.privacy_beg_id:
                    trigger_privacy = True
                elif token_id == self.privacy_end_id:
                    trigger_privacy = False
                else:
                    new_input_id.append(token_id)
                    if trigger_privacy:
                        mask_pos_list.append(idx)
                    idx += 1
            seq_len = len(new_input_id)
            seq_len_list.append(seq_len)

            attention_mask = torch.tril(torch.ones((seq_len, seq_len)))
            for idx in mask_pos_list:
                # The last token can access everything.
                attention_mask[idx+1:split_idx, idx] = 0
                attention_mask[idx,:idx] = 1
            new_attention_masks.append(attention_mask)
            new_input_ids.append(new_input_id)

            max_len = max(max_len, seq_len)
        
        new_full_attention_mask = torch.zeros((batch_size, max_len))
        for batch, seq_len in enumerate(seq_len_list):
            if self.padding_side == 'left':
                new_full_attention_mask[batch, max_len-seq_len:] = 1
            else:
                new_full_attention_mask[batch, :seq_len] = 1

        for idx, (input_ids, attention_mask) in enumerate(zip(new_input_ids, new_attention_masks)):
            current_len = len(input_ids)
            new_attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            if self.padding_side == 'left':
                input_ids = [self.pad_token_id]*(max_len - current_len) + input_ids
            else:
                input_ids = input_ids + [self.pad_token_id]*(max_len - current_len)
            
            if self.padding_side == 'left':
                new_attention_mask[max_len-current_len:, max_len-current_len:] = attention_mask
            else:
                new_attention_mask[:current_len,:current_len] = attention_mask
            new_input_ids[idx] = torch.tensor(input_ids).unsqueeze(0)
            new_attention_masks[idx] = new_attention_mask.unsqueeze(0)
        input_id = torch.cat(new_input_ids, dim=0)
        attention_mask = torch.cat(new_attention_masks, dim=0)
        position_ids = new_full_attention_mask.cumsum(-1) - 1
        return {'input_ids': input_id, 'attention_mask': attention_mask, "full_attention_mask": new_full_attention_mask,"position_ids": position_ids}


if __name__ == '__main__':
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    custom_tokenizer = CustomLlamaTokenizer.from_pretrained(model_id)
    custom_tokenizer.add_privacy_tokens()

    # Tokenize some text with the custom tokenizer
    example_text_0 = "This is <privacy>an example text with special tokens</privacy> does it work?"
    example_text_1 = "This is <privacy>me</privacy>, okay?"
    example_text_2 = "This is <privacy>me</privacy>, okay?"
    encoding = custom_tokenizer([example_text_0, example_text_1, example_text_2], split_idx_list=[-1,-1,-2])

    # Inspect the encoding
    print(encoding)