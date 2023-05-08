"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from scripts.oasst_utils import load_oasst_export
from tqdm import tqdm


DATA_FILE_NAME = "2023-04-12_oasst_ready.trees.jsonl.gz"
IGNORE_INDEX = -1

def prepare(
    destination_path: Path = Path("data/oasst"), 
    tokenizer_path: Path = Path("/Users/zhenyishen/Downloads/LLAMA-tokenizer/tokenizer.model"),
    test_split_ratio: float = 0.05,
    max_seq_length: int = 1024,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    destination_path.mkdir(parents=True, exist_ok=True)
    file_path = destination_path / DATA_FILE_NAME
    #download(file_path)

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    
    train_set, test_set = load_oasst_export(
            file_path,
            val_split=test_split_ratio,
            manual_seed=seed,
            lang="bg,ca,cs,da,de,en,es,fr,hr,hu,it,nl,pl,pt,ro,ru,sl,sr,sv,uk",
            mode= "sft",
        )

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    torch.save(train_set, file_path.parent / "train.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)]
    torch.save(test_set, file_path.parent / "test.pt")


def prepare_sample(conversations: list, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True, truncated_stats=[0]):
    """Processes a single sample.
    
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string
    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.
    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    encoded_conversations = []
    formatted_conversations = []
    for i, converse in enumerate(conversations):
        if i%2==0: # human
            role = "Human"
        else:
            role = "Assistant"
        
        bos = True if i == 0 else False

        encoded_conversations.append(tokenize(
            tokenizer, f"{role}: {converse}", max_length=max_length, bos=bos
        ))
        formatted_conversations.append(f"{role}: {converse}")

    encoded_conversations = torch.cat(encoded_conversations)
    if encoded_conversations.size(0) > max_length:
        truncated_stats[0] += 1
        encoded_conversations = encoded_conversations[:max_length]
        print("truncated_stats", truncated_stats[0])

    # calculate len of prompt
    encoded_full_prompt = tokenize(tokenizer, conversations[0], max_length=max_length, bos=True)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_conversations.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {"conversations": formatted_conversations, "input_ids": encoded_conversations, "labels": labels, }


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, bos: bool = True) -> torch.Tensor:
    return tokenizer.encode(string, bos=bos, eos=True, max_length=max_length)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)

