from lit_llama.tokenizer import Tokenizer
import torch
tokenizer_path = '/Users/zhenyishen/Downloads/LLaMA/tokenizer.model'
tokenizer = Tokenizer(tokenizer_path)

print(tokenizer.eos_id)
print(tokenizer.decode(torch.Tensor([234, tokenizer.eos_id], dtype=torch.int64)))