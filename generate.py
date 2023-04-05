import time
from pathlib import Path

import torch
from torch import nn
from dataclasses import dataclass

import lightning as L
from lit_llama import LLaMA, Tokenizer

"""paths and parameters"""
prompt = "here is "
accelerator = 'cpu'
tokenizer_path = "data/shakespeare/tokenizer.model"#"/Users/zhenyishen/Downloads/LLaMA/tokenizer.model"
model_path = "out/training/latest-iter-ckpt.pt"

tokenizer_path = Path(tokenizer_path)
model_path = Path(model_path)

def generate(
    model: nn.Module,
    tokens: torch.Tensor,
    max_new_tokens: int = 16,
    max_length: int = 128,
    temperature: float = 0.8,
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

    fabric = L.Fabric(accelerator=accelerator)

    print("Loading the checkpoint ...")
    t0 = time.time()
    model = LLaMA.from_name("test")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
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
    main()