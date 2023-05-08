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
from tqdm import tqdm


DATA_FILE_NAME = "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
IGNORE_INDEX = -1


def prepare(
    destination_path: Path = Path("/Users/zhenyishen/Downloads"), 
    tokenizer_path: Path = Path("/Users/zhenyishen/Downloads/LLAMA-tokenizer/tokenizer.model"),
    test_split_size: int = 2000,
    max_seq_length: int = 2048,
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
    
    with open(file_path, "r") as file:
        data = json.load(file)

    # Partition the dataset into train and test
    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data, 
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    proc_train_set = []
    for sample in tqdm(train_set):
        try:
            proc_train_set.append(prepare_sample(sample, tokenizer, max_seq_length, mask_inputs))
        except ValueError:
            continue
        except RuntimeError:
            continue
    torch.save(proc_train_set, file_path.parent / "train.pt")

    print("Processing test split ...")
    proc_test_set = []
    for sample in tqdm(test_set):
        try:
            proc_test_set.append(prepare_sample(sample, tokenizer, max_seq_length, mask_inputs))
        except ValueError:
            continue
        except RuntimeError:
            continue
    torch.save(proc_test_set, file_path.parent / "test.pt")


def download(file_path: Path):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w") as f:
        f.write(requests.get(DATA_FILE).text)


def prepare_sample(conversations: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True, truncated_stats=[0]):
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
    for i, item in enumerate(conversations["conversations"]):
        if len(item) == 0:
            raise ValueError
        role = item["from"]
        converse = item["value"]

        bos = True if i == 0 else False

        encoded_conversations.append(tokenize(
            tokenizer, f"{role}: {converse}", max_length=max_length, bos=bos
        ))
        formatted_conversations.append(f"{role}: {converse}")

    #print(conversations["conversations"])
    #print(encoded_conversations)
    encoded_conversations = torch.cat(encoded_conversations)

    if encoded_conversations.size(0) > max_length:
        truncated_stats[0] += 1
        encoded_conversations = encoded_conversations[:max_length]
        print("truncated_stats", truncated_stats[0])

    # calculate len of prompt
    encoded_full_prompt = tokenize(tokenizer, conversations["conversations"][0]["value"], max_length=max_length, bos=True)

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