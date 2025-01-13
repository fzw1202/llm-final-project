from prefill import Prefill
from decode import Decode
from transformers import GPT2Tokenizer
import torch
import numpy as np

def add_new_ids(tokenizer, input_ids, attention_mask, new_input_ids, device):
    new_id_column = torch.full((input_ids.size(0), 1), tokenizer.pad_token_id, device=device)
    new_mask_column = torch.full((attention_mask.size(0), 1), 0, device=device)
    input_ids = torch.cat((input_ids, new_id_column), dim=1)
    attention_mask = torch.cat((attention_mask, new_mask_column), dim=1)
    last_positions = np.argmax(attention_mask.to('cpu') == 0, axis=1)
    for index, element in enumerate(last_positions):
        input_ids[index, element] = new_input_ids[index]
        attention_mask[index, element] = 1
    return input_ids, attention_mask

class Scheduler:
    def __init__(self, batch_size, max_context_token_num, stop_words):
        self.batch_size = batch_size
        self.device = 'cuda:1'
        self.prefillModel = Prefill('cuda:0', self.device)
        self.decodeModel = Decode('cuda:1', self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_docode_token_num = max_context_token_num
        self.stop_words_token = self.tokenizer.convert_tokens_to_ids(stop_words)

    def process(self, prompts):
        first_batch = prompts[:self.batch_size]
        left_prompts = prompts[self.batch_size:]
        return self.continuous_batch(first_batch, left_prompts)

    def getDevice(self):
        return self.device

    def encode(self, prompts, seq_len):
        encodeBatch = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encodeBatch['input_ids'].to(self.getDevice())
        padding_len = seq_len - input_ids.size(1)
        input_ids = torch.cat((input_ids, torch.full((input_ids.size(0), padding_len), self.tokenizer.pad_token_id, device=self.getDevice())), dim=-1)
        attention_mask = encodeBatch['attention_mask'].to(self.getDevice())
        attention_mask = torch.cat((attention_mask, torch.full((attention_mask.size(0), padding_len), 0, device=self.getDevice())), dim=-1)
        return input_ids, attention_mask

    def idPadding(self, input_ids, seq_len):
        padding_len = seq_len - input_ids.size(1)
        return torch.cat((input_ids, torch.full((input_ids.size(0), padding_len), self.tokenizer.pad_token_id, device=self.getDevice())), dim=-1)

    def maskPadding(self, attention_mask, seq_len):
        padding_len = seq_len - attention_mask.size(1)
        return torch.cat((attention_mask, torch.full((attention_mask.size(0), padding_len), 0, device=self.getDevice())), dim=-1)

    def kvcachePadding(self, kv, seq_len):
        batch_size, origin_seq_len, embed_dim = kv[0][0].size()
        padding_len = seq_len - origin_seq_len
        key_padding = torch.full((batch_size, padding_len, embed_dim), 0, device=self.getDevice())
        value_padding = torch.full((batch_size, padding_len, embed_dim), 0, device=self.getDevice())
        return [(torch.cat((kv_item[0], key_padding), dim=1), torch.cat((kv_item[1], value_padding), dim=1)) for kv_item in kv]

    def encodePrompts(self, prompts):
        encodeBatch = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        return encodeBatch['input_ids'].to(self.getDevice()), encodeBatch['attention_mask'].to(self.getDevice())

    def continuous_batch(self, first_batch_prompts, left_prompts):
        input_ids, mask = self.encodePrompts(first_batch_prompts)
        new_ids, kv = self.prefillModel(input_ids, mask)
        input_ids, mask = add_new_ids(self.tokenizer, input_ids, mask, new_ids, device=self.getDevice())
        generate_result = []

        while input_ids.size(0) > 0:
            new_ids, kv = self.decodeModel(input_ids, kv, mask)
            input_ids, mask = add_new_ids(self.tokenizer, input_ids, mask, new_ids, device=self.getDevice())
            seq_len = input_ids.size(1)
            stop_decode_indexs = [
                index for index, (new_id, mask) in enumerate(zip(new_ids, mask))
                if new_id in self.stop_words_token or mask.sum() > self.max_docode_token_num
            ]

            delete_num = 0
            for index in stop_decode_indexs:
                fix_index = index - delete_num
                generate_result.append(input_ids[fix_index])
                input_ids = torch.cat((input_ids[:fix_index], input_ids[fix_index + 1:]), dim=0)
                mask = torch.cat((mask[:fix_index], mask[fix_index + 1:]), dim=0)
                kv = [(torch.cat((kv_item[0][:fix_index], kv_item[0][fix_index + 1:]), dim=0),
                       torch.cat((kv_item[1][:fix_index], kv_item[1][fix_index + 1:]), dim=0))
                      for kv_item in kv]
                delete_num += 1

            if len(left_prompts) > 0 and len(stop_decode_indexs) > 0:
                prefill_num = len(stop_decode_indexs)
                seq_len = mask.sum(dim=1).max() if input_ids.size(0) > 0 else 0
                prefill_ids, prefill_mask = self.encodePrompts(left_prompts[:prefill_num])

                if seq_len < input_ids.size(1):
                    reduced_len = input_ids.size(1) - seq_len
                    input_ids = input_ids[:, :-reduced_len]
                    mask = mask[:, :-reduced_len]
                    kv = [(kv_item[0][:, :-reduced_len], kv_item[1][:, :-reduced_len]) for kv_item in kv]

                if seq_len > prefill_ids.size(1):
                    prefill_ids, prefill_mask = self.encode(left_prompts[:prefill_num], seq_len - 1)
                    left_prompts = left_prompts[prefill_num:]
                    new_prefill_ids, new_prefill_kv = self.prefillModel(prefill_ids, prefill_mask)
                    new_ids, new_mask = add_new_ids(self.tokenizer, prefill_ids, prefill_mask, new_prefill_ids, device=self.getDevice())
                    input_ids = torch.cat((input_ids, new_ids), dim=0)
                    mask = torch.cat((mask, new_mask), dim=0)
                    kv = [(torch.cat((old_kv[0], new_kv[0]), dim=0), torch.cat((old_kv[1], new_kv[1]), dim=0)) for old_kv, new_kv in zip(kv, new_prefill_kv)]
                else:
                    new_prefill_ids, new_prefill_kv = self.prefillModel(prefill_ids, prefill_mask)
                    new_ids, new_mask = add_new_ids(self.tokenizer, prefill_ids, prefill_mask, new_prefill_ids, device=self.getDevice())
                    left_prompts = left_prompts[prefill_num:]
                    if seq_len != 0:
                        new_seq_len = new_ids.size(1)
                        input_ids = self.idPadding(input_ids, new_seq_len)
                        mask = self.maskPadding(mask, new_seq_len)
                        kv = self.kvcachePadding(kv, new_seq_len - 1)
                        input_ids = torch.cat((input_ids, new_ids), dim=0)
                        mask = torch.cat((mask, new_mask), dim=0)
                        kv = [(torch.cat((old_kv[0], new_kv[0]), dim=0), torch.cat((old_kv[1], new_kv[1]), dim=0)) for old_kv, new_kv in zip(kv, new_prefill_kv)]
                    else:
                        input_ids = new_ids
                        mask = new_mask
                        kv = new_prefill_kv

        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generate_result]
