import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import os
import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm


IGNORE_INDEX = -1

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def get_examples(data_dir, split):
    path = os.path.join(data_dir, f"{split}.jsonl")
    examples = read_jsonl(path)

    print(f"{len(examples)} {split} examples")
    return examples

def prepare(
    destination_path: Path = Path("data/GSM8K"), 
    tokenizer_path: Path = Path("/Users/zhenyishen/Downloads/LLAMA-tokenizer/tokenizer.model"),
    max_seq_length: int = 1024,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    destination_path.mkdir(parents=True, exist_ok=True)
    #download(file_path)

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    
    train_data = read_jsonl(destination_path / "train.jsonl")
    test_data = read_jsonl(destination_path / "test.jsonl")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_data)]
    train_set = [x for x in train_set if x is not None]

    test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_data)]
    test_set = [x for x in test_set if x is not None]

    print(f"training data has {len(train_set)} samples")
    print(f"test data has {len(test_set)} samples")

    torch.save(train_set, destination_path / "train.pt")
    torch.save(test_set, destination_path / "test.pt")


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
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
    full_prompt = generate_prompt(example)
    answer = example["answer"].replace("\n#### ", "\nThe answer is ")
    full_prompt_and_response = full_prompt + answer
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, max_length=max_length)
    if encoded_full_prompt_and_response.size(0) == max_length:
        return None

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    ##print("full_prompt_and_response", full_prompt_and_response)
    #print("encoded_full_prompt_and_response", encoded_full_prompt_and_response.shape)
    return {**example, "input_ids": encoded_full_prompt_and_response, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=True, max_length=max_length)


def generate_prompt(example):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction: \n{example['question']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)