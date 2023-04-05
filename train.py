import os
import time
from functools import partial
from typing import Tuple

import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import numpy as np

import lightning as L
from lightning.fabric.strategies import FSDPStrategy

from lit_llama.model import Block, LLaMA, LLaMAConfig
from lit_llama.utils import save_model_checkpoint

debug=True
accelerator='cpu'
n_devices = 1
precision="16-mixed"

out_dir = "out/training"
eval_interval = 2000
eval_iters = 200
log_interval = 5
# compilation fails as it does not support torch.complex64 for RoPE
# compile = False

# Hyperparameters
learning_rate = 6e-4
batch_size = 2
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# For shakespeare, choose smaller block size than vanilla LLaMA
block_size = 1024

def main() -> None:
    if not debug:
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block)
        fabric = L.Fabric(accelerator=accelerator, devices=n_devices, precision=precision, strategy=strategy)
    else:
        fabric = L.Fabric(accelerator=accelerator)

    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig.from_name("test")
    config.block_size = block_size
    config.vocab_size = 100  # from prepare_shakespeare.py

    #with fabric.device:
    model = LLaMA(config)

    model = fabric.setup_module(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    optimizer = fabric.setup_optimizers(optimizer)

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

    iter_num = 0

    while True:
        if iter_num > 0 and iter_num % eval_interval == 0:
            val_loss = validate(fabric, model, val_data)
            fabric.print(f"iter {iter_num}: val_loss {val_loss:.4f}")
            fabric.print(f"Saving checkpoint to {out_dir}")
            save_model_checkpoint(fabric, model, os.path.join(out_dir, f"latest-iter-ckpt.pt"))

        t0 = time.time()

        input_ids, targets = get_batch(
            fabric,
            train_data,
            model.config.block_size
        )
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        fabric.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

        t1 = time.time() - t0
        
        if iter_num > 0 and iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time {t1*1000:.2f}ms")

        iter_num += 1

@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(
            fabric,
            val_data,
            model.config.block_size
        )
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        losses[i] = loss
    mean_loss = losses.mean()
    model.train()
    return mean_loss


def get_batch(fabric: L.Fabric, data: np.ndarray, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix
    ])
    if not debug:
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir: str = "data/shakespeare") -> Tuple[np.ndarray, np.ndarray]:
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train_data, val_data

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    main()