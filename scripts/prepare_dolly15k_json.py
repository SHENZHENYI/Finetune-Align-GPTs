import re
import json
from datasets import load_dataset

# @agoryuno contributed this
re_reference_remove = re.compile(r"\[\d+(?:,\s*\d+)*?\]")
re_single_reference_remove = re.compile(r"\[\s?\d+\s?\]")
citation_regex = re.compile(r"\[[a-zA-Z]\]")
# check if the whole string is just a combination of (multiple) whitespaces and newlines
re_whitespace_newline_match = re.compile(r"^[\s\n]*$")


def _process_instruction(row, input_max_length):
    context = re_reference_remove.sub("", row["METADATA"]["CONTEXT"])
    # further remove references
    context = context.replace("[citation needed]", "")
    context = citation_regex.sub("", context)
    return {
        "context": context,
        "questions": row["INSTRUCTION"][:input_max_length],
        "answers": row["RESPONSE"][:input_max_length],
    }

cache_dir = "/Users/zhenyishen/Documents/GitHub/Finetune-Align-GPTs/data/dolly"
data = load_dataset("OllieStanley/oa_dolly_15k", cache_dir=cache_dir)

train_data = [_process_instruction(x, input_max_length=1024) for x in data["train"]]

data = train_data

print(data[0])

with open("/Users/zhenyishen/Documents/GitHub/Finetune-Align-GPTs/data/dolly/dolly-15k-cleaned.json", 'w') as fp:
    json.dump(data, fp, indent=6)
