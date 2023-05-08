import re
import json
from datasets import load_dataset

def _process_instruction(row):
    return {
        "INSTRUCTION": row["INSTRUCTION"],
        "RESPONSE": row["RESPONSE"],
    }

data = load_dataset("qwedsacf/grade-school-math-instructions")['train']

data = [_process_instruction(x) for x in data]

with open("/Users/zhenyishen/Documents/GitHub/Finetune-Align-GPTs/data/grademath/grade-school-math-instructions.json", 'w') as fp:
    json.dump(data, fp, indent=6)
