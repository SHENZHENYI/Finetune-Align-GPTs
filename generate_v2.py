import time
from pathlib import Path

import torch
from torch import nn
from dataclasses import dataclass

import lightning as L
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice
from lit_llama.model import LLaMA, LLaMAConfig
from prompts import PromptTemplate


"""paths and parameters"""
prompt_tmp = PromptTemplate()
prompt = prompt_tmp.construct_prompt(question="Tell me about the president of Mexico in 2019.")
use_lora = True
llama_name = "7B"
accelerator = 'cuda'
n_devices=1
tokenizer_path = "../tokenizer.model"#"/Users/zhenyishen/Downloads/LLaMA/tokenizer.model"
model_path = "../state_dict.pth"
lora_path = "../alpaca-lora-512/iter-074239-ckpt.pt"
block_size=512
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05


tokenizer_path = Path(tokenizer_path)
model_path = Path(model_path)

def generate(
    model: nn.Module,
    tokens: torch.Tensor,
    max_new_tokens: int = 128,
    max_length: int = 256,
    temperature: float = 1.0,
    top_k: int = 200,
    stop_token: int = 2
):
    B, T = tokens.size()
    T_new = T + max_new_tokens
    tmp = torch.empty(B, T_new, dtype=tokens.dtype, device=tokens.device)
    stop_len = T_new
    tmp[:, :T] = tokens
    tokens = tmp  
    #print('stop_len', stop_len, 'T_new', T_new)

    for t in range(T, T_new):
        tokens_conditional = tokens[:, :t]
        tokens_conditional = tokens_conditional if t < max_length else tokens_conditional[:, -max_length:]

        logits = model(tokens_conditional)
        logits = logits[:, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).type(tokens.dtype)

        tokens[:, [t]] = next_token

        if next_token.item() == stop_token:
            stop_len = t+1
            break
            

    return tokens[:, :stop_len]

def main():
    assert tokenizer_path.is_file()
    assert model_path.is_file()

    fabric = L.Fabric(accelerator=accelerator, devices=n_devices)

    dtype = None
    if dtype is not None:
        dt = getattr(torch, dtype, None)
        if not isinstance(dt, torch.dtype):
            raise ValueError(f"{dtype} is not a valid dtype.")
        dtype = dt

    config = LLaMAConfig.from_name(llama_name)
    config.block_size = block_size
    checkpoint = torch.load(model_path)
    
    if use_lora:
        print('lora')
        lora_checkpoint = torch.load(lora_path)
        with fabric.device, lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            torch.set_default_tensor_type(torch.HalfTensor)
            model = LLaMA(config).bfloat16()
            torch.set_default_tensor_type(torch.FloatTensor)
            # strict=False because missing keys due to LoRA weights not contained in checkpoint state
            model.load_state_dict(checkpoint, strict=False) 
            model.load_state_dict(lora_checkpoint, strict=False) 

    else:
        print('no lora')
        with fabric.device:
            torch.set_default_tensor_type(torch.HalfTensor)
            model = LLaMA(config).bfloat16()
            torch.set_default_tensor_type(torch.FloatTensor)
            # strict=False because missing keys due to LoRA weights not contained in checkpoint state
            model.load_state_dict(checkpoint, strict=False) 
        
    model.eval()
    model = fabric.setup_module(model)

    # encode
    tokenizer = Tokenizer(tokenizer_path)
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)
    encoded_prompt = encoded_prompt[None, :]  # add batch dimension
    
    L.seed_everything(1234)

    tokens = generate(model, encoded_prompt, stop_token=tokenizer.eos_id, max_new_tokens=block_size, max_length= block_size,)[0]
    print(tokens)
    print(tokens.shape)
    out = tokenizer.decode(tokens)
    print(out)

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    main()