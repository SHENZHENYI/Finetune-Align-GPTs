"""
Instruction-tuning with LoRA on the Alpaca dataset.
Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", install
the PyTorch nightly version for a fix (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import os
import time

import lightning as L
import numpy as np
import torch

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.utils import EmptyInitOnDevice
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt

debug = False
llama_name = "7B"
load_from_the_pretrained = True
model_path = "../state_dict.pth"
tokenizer_path = "../tokenizer.model"

out_dir = "out/alpaca-lora"
eval_interval = 20
save_interval = 20
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 3e-4
batch_size = 128
micro_batch_size = 6
gradient_accumulation_steps = batch_size // micro_batch_size
max_iters = 50000 * 3 // micro_batch_size
weight_decay = 0.0
block_size = 256
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_steps = 100

def main():
    if not debug:
        fabric = L.Fabric(accelerator="cuda", devices=2, precision="bf16-mixed", strategy='ddp')
    else:
        fabric = L.Fabric(accelerator="cpu",)
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig.from_name(llama_name)
    config.block_size = block_size

    with EmptyInitOnDevice(device=fabric.device, dtype=torch.bfloat16):
        with lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA(config)

    if load_from_the_pretrained:
        checkpoint = torch.load(model_path)
        
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False) 

    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> None:
    """The training loop.
    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0

    for iter_num in range(max_iters):
        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        fabric.backward(loss)

        #fabric.clip_gradients(model, optimizer, clip_val=1.0)

        if (iter_num + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
                
            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pt"), checkpoint)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")

def generate_response(model, instruction):
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=True)
    encoded = encoded[None, :]  # add batch dimension
    encoded = encoded.to(model.device)

    output = generate(
        model,
        tokens=encoded,
        max_length=block_size,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output[0].cpu())
    return output # output.split("### Response:")[1].strip()

@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, instruction)
    fabric.print(instruction)
    fabric.print(output)

    model.train()
    return out.item()

def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size, ))

    input_ids = [torch.tensor(data[i]["input_ids"], dtype=torch.int64) for i in ix]
    labels = [torch.tensor(data[i]["labels"], dtype=torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_left(x, pad_id):
        n = max_len-len(x)
        return torch.cat((
            torch.full((n,), pad_id, dtype=x.dtype),
            x
        ))
    
    x = torch.stack([pad_left(xx, pad_id=0) for xx in input_ids])
    y = torch.stack([pad_left(yy, pad_id=-1) for yy in labels])

    # shift the input and the targets
    x = x[:, :-1]
    y = y[:, 1:]

    if not debug:
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))

    return x, y


def load_datasets(data_dir: str = "data/alpaca"):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
