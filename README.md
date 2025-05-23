# LagKV

#### Introduction

LagKV is an efficient and robust KV compression algorithm. It uses lag tokens information to compress the previous ones which significantly boost the compression performance with little computation overhead.

Details are in the following work:
** [LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important](https://arxiv.org/abs/2504.04704) **

#### How to Use

LagKV implements the Cache interface from transformers. It's easy to be integrated into the model calling function.

```python
from lag_kv import LagKV
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Qwen2.5-7B-Instruct"
device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", attn_implementation="sdpa").to(device)

prompt = "long text"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
past_key_values = LagKV(lag_size=64)
print(model.generate(input_ids, past_key_values=past_key_values))
# check KV cache size
print(past_key_values[0][0].size())
```

To compress the KV cache during the prefill stage instead of it's precisely calculated, you have to use the following inference function(for batch_size=1 only.):

```python
def inference_by_prefill_compress(model, tokenizer, inputs, max_new_tokens=256, decode=False, past_key_values=None, device="cuda"):
    if isinstance(inputs, str):
        input_ids = tokenizer([inputs], return_tensors="pt")["input_ids"].to(device)
    else:
        input_ids = inputs
    if past_key_values is None:
        past_key_values = LagKV(ratio=0.2,
                             lag_size=128,
                            layer_idx_skip_first=[],
                             use_then_compress=True)
    
    with torch.no_grad():
        sink_size = past_key_values.sink_size
        lag_size = past_key_values.lag_size
        trigger_len = sink_size + 2*lag_size
        input_length = input_ids.shape[1]
        # print(input_length > trigger_len)
        if input_length > trigger_len:
            start_idx = 0
            end_idx = trigger_len
            position_ids = torch.arange(input_length + max_new_tokens).unsqueeze(0).to(device)
            def batch_input():
                sel_input_ids = input_ids[:, start_idx:end_idx]
                q_len = end_idx - start_idx
                k_len = past_key_values.get_seq_length() + q_len
                batch_size = input_ids.shape[0]
                head_num = model.config.num_attention_heads
                attn_mask = torch.ones((k_len, q_len), 
                							device=input_ids.device, dtype=torch.bool)
                attn_mask = torch.triu(attn_mask, diagonal=1).T
                attn_mask = torch.flip(attn_mask, (0, 1))
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_mask = attn_mask.expand(batch_size, -1, -1, -1).expand(-1, head_num, -1, -1)
                attention_mask = torch.zeros((batch_size, head_num, q_len, k_len), device=input_ids.device, dtype=torch.bfloat16)
                attention_mask.masked_fill_(attn_mask, -torch.inf)
                return {"input_ids": sel_input_ids, "attention_mask": attention_mask}
            
            while start_idx < input_length:
                tmp_pos = position_ids[:, start_idx:end_idx]
                outputs = model(**batch_input(), 
                               past_key_values=past_key_values,
                              position_ids=tmp_pos,
                              cache_position=tmp_pos[0]
                              )
                start_idx = end_idx
                end_idx += lag_size
                end_idx = min(end_idx, input_length)

            new_token_id = outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)
            # print(new_token_id)
            new_token_count = 1
            generated_ids = [new_token_id]
            while new_token_id[0][0] != tokenizer.eos_token_id and new_token_count < max_new_tokens+1:
                tmp_pos = position_ids[:, (input_length+new_token_count-1):(input_length+new_token_count)]
                outputs = model(new_token_id, 
                               past_key_values=past_key_values,
                              position_ids=tmp_pos,
                              cache_position=tmp_pos[0]
                              )
                new_token_id = outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)
                new_token_count += 1
                generated_ids.append(new_token_id)
            generated_ids = torch.cat(generated_ids, dim=-1)
        else:
            generated_ids = model.generate(inputs, do_sample=False, max_new_tokens=max_new_tokens, past_key_values=past_key_values)
            generated_ids = generated_ids[:, input_length:]
    if decode:
        output = tokenizer.batch_decode(generated_ids)
    else:
        output = generated_ids
    return output, past_key_values
```