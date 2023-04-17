from typing import List

class PromptTemplate:
    instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction: {question}\n### Response:\n"
    question: str 

    def construct_prompt(self, question: str, instruction=None):
        if instruction is not None:
            return instruction.replace('{question}', question)
        return self.instruction.replace('{question}', question)