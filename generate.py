import time
from pathlib import Path

import torch
from torch import nn
from dataclasses import dataclass

import lightning as L
from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice
from lit_llama.model import LLaMA, LLaMAConfig


"""paths and parameters"""
prompt = "here is "
llama_name = "7B"
accelerator = 'cuda'
n_devices=1
tokenizer_path = "../tokenizer.model"#"/Users/zhenyishen/Downloads/LLaMA/tokenizer.model"
model_path = "../state_dict.pth"
block_size=256

tokenizer_path = Path(tokenizer_path)
model_path = Path(model_path)

def generate(
    model: nn.Module,
    tokens: torch.Tensor,
    max_new_tokens: int = 128,
    max_length: int = 256,
    temperature: float = 1.0,
    top_k: int = 200
):
    B, T = tokens.size()
    T_new = T + max_new_tokens
    tmp = torch.empty(B, T_new, dtype=tokens.dtype, device=tokens.device)
    tmp[:, :T] = tokens
    tokens = tmp  

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

    return tokens

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
    with fabric.device:
        t0 = time.time()
        torch.set_default_tensor_type(torch.HalfTensor)
        model = LLaMA(config).bfloat16()
        torch.set_default_tensor_type(torch.FloatTensor)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False) 
        t1 = time.time()-t0
        print(f"Loaded the checkpoint: {t1:.4f}s")

    model.eval()
    model = fabric.setup_module(model)

    # encode
    tokenizer = Tokenizer(tokenizer_path)
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)
    encoded_prompt = encoded_prompt[None, :]  # add batch dimension
    
    L.seed_everything(1234)

    tokens = generate(model, encoded_prompt)[0]
    out = tokenizer.decode(tokens)
    print(out)

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    main()