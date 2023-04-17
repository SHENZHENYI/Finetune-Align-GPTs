"""
evaluate on GSM8K
first-eval: 38/527 (not complete)
"""

import time
from pathlib import Path
import os
import json
import re

import torch
from torch import nn
from dataclasses import dataclass

import lightning as L
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice
from lit_llama.model import LLaMA, LLaMAConfig
from prompts import PromptTemplate


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def get_examples(data_dir, split):
    path = os.path.join(data_dir, f"{split}.jsonl")
    examples = read_jsonl(path)

    print(f"{len(examples)} {split} examples")
    return examples

def extract_answer(answer: str):
    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, answer)
    if len(pred) >= 1:
        return pred[-1]
    return float('inf') # a default False -- hacking

"""paths and parameters"""
prompt_tmp = PromptTemplate()

#tmp1 = 'Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: 72'
#tmp2 = 'Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: 10'
#tmp3 = 'Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nAnswer: 5'

tmp = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
"""

instruction = f"""Follow the given examples and answer the question.\n{tmp}\nQuestion: """ +  "{question}\nAnswer: Let's think step by step. "
use_lora = True
llama_name = "7B"
accelerator = 'cuda'
n_devices=1
tokenizer_path = "../tokenizer.model"#"/Users/zhenyishen/Downloads/LLaMA/tokenizer.model"
model_path = "../state_dict.pth"
lora_path = "./alpaca-lora-512/iter-074239-ckpt.pt"
block_size=512
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05

tokenizer_path = Path(tokenizer_path)
model_path = Path(model_path)

test_data = get_examples("./data/math", "test")


def generate(
    model: nn.Module,
    tokens: torch.Tensor,
    tokenizer,
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

    if temperature == 0.: # hacking
        temperature = 1.0
        top_k = 1

    answer_str = ''
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

        answer_str += tokenizer.decode(next_token)[0]

        #print(tokenizer.decode(next_token))

        #if next_token.item() == stop_token or answer_str.endswith("Question"):
        if answer_str.endswith("Question"):
            stop_len = t+1
            break

    return tokens[:, T:stop_len]

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

    n_correct = 0
    n_count = 0
    for test_sample in test_data:
        q = test_sample['question']
        a = test_sample['answer']
        n_count += 1
        prompt = prompt_tmp.construct_prompt(question=q, instruction=instruction)
        encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)
        encoded_prompt = encoded_prompt[None, :]  # add batch dimension
        
        #L.seed_everything(1234)

        tokens = generate(model, encoded_prompt, tokenizer, stop_token=tokenizer.eos_id, max_new_tokens=block_size, max_length= block_size, temperature=0.)[0]
        #print(tokens)
        #print(tokens.shape)
        out = tokenizer.decode(tokens)
        #print(out)
        print('====cot: ', out)
        #if 'The answer is ' not in out:
        #    print(f'gt={a}; pred=invalid; n_correct={n_correct}; count={n_count}')
        #    continue
        pred_answer = extract_answer(out)
        gt_answer = extract_answer(a)

        print("===pred_answer", pred_answer)
        print("===gt_answer", gt_answer)

        if pred_answer == gt_answer:
            n_correct += 1
        print(f'gt={gt_answer}; pred={pred_answer}; n_correct={n_correct}; count={n_count}')
        print('-------------------')

    print(n_correct)

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    main()