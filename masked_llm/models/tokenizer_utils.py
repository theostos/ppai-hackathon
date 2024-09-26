from typing import List

from transformers import AutoTokenizer
import torch

def prepare_tokenizer(tokenizer: AutoTokenizer, token_beg="<sensitive>", token_end="</sensitive>"):
    """
    Add privacy special tokens
    """
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [token_beg, token_end]})
    tokenizer.sensitive_beg_id = tokenizer.encode(token_beg, add_special_tokens=False)[0]
    tokenizer.sensitive_end_id = tokenizer.encode(token_end, add_special_tokens=False)[0]

def generate_custom_mask(tokenizer: AutoTokenizer, prompts: List[str], device='cpu'):
    """
    Given a prepared tokenizer (i.e. with privacy special tokens), a list of prompts with privacy special tokens,
    tokenize and generate custom masks for a privacy-compatible transformer.
    """
    input_ids = tokenizer(prompts)['input_ids']
    return generate_custom_mask_input_ids(tokenizer, input_ids, device=device)[0]

def generate_custom_mask_input_ids(tokenizer: AutoTokenizer, input_ids, device='cpu', padding_side="right"):
    """
    Given a prepared tokenizer (i.e. with privacy special tokens), a list of prompts with privacy special tokens,
    tokenize and generate custom masks for a privacy-compatible transformer.
    """
    new_input_ids, new_attention_masks, seq_len_list = [], [], []
    max_len = 0
    batch_size = len(input_ids)
    for input_id in input_ids:
        trigger_privacy = False
        new_input_id = []
        mask_pos_list = []
        idx = 0
        for token_id in input_id:
            if token_id == tokenizer.sensitive_beg_id:
                trigger_privacy = True
            elif token_id == tokenizer.sensitive_end_id:
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
            attention_mask[idx+1:-1, idx] = 0
            attention_mask[idx,:idx] = 1
        new_attention_masks.append(attention_mask)
        new_input_ids.append(new_input_id)

        max_len = max(max_len, seq_len)
    
    new_full_attention_mask = torch.zeros((batch_size, max_len))
    for batch, seq_len in enumerate(seq_len_list):
        if padding_side == 'left':
            new_full_attention_mask[batch, max_len-seq_len:] = 1
        else:
            new_full_attention_mask[batch, :seq_len] = 1

    for idx, (input_ids, attention_mask) in enumerate(zip(new_input_ids, new_attention_masks)):
        current_len = len(input_ids)
        new_attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        if padding_side == 'left':
            input_ids = [tokenizer.pad_token_id]*(max_len - current_len) + input_ids
        else:
            input_ids = input_ids + [tokenizer.pad_token_id]*(max_len - current_len)
        
        if padding_side == 'left':
            new_attention_mask[max_len-current_len:, max_len-current_len:] = attention_mask
        else:
            new_attention_mask[:current_len,:current_len] = attention_mask
        new_input_ids[idx] = torch.tensor(input_ids).unsqueeze(0)
        new_attention_masks[idx] = new_attention_mask.unsqueeze(0)
    input_id = torch.cat(new_input_ids, dim=0)
    attention_mask = torch.cat(new_attention_masks, dim=0)
    return {'input_ids': input_id.to(device), 'attention_mask': attention_mask.to(device)}, new_full_attention_mask.to(device)