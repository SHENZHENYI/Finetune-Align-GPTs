from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from torch import Generator
from torch.utils.data import Dataset

import gzip
import json
from pathlib import Path
from typing import Callable, Iterable, Optional, TextIO

import pydantic

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset

def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

class LabelAvgValue(BaseModel):
    value: float | None
    count: int


LabelValues = dict[str, LabelAvgValue]


class ExportMessageEvent(BaseModel):
    type: str
    user_id: str | None


class ExportMessageEventEmoji(ExportMessageEvent):
    type: Literal["emoji"] = "emoji"
    emoji: str


class ExportMessageEventRating(ExportMessageEvent):
    type: Literal["rating"] = "rating"
    rating: str


class ExportMessageEventRanking(ExportMessageEvent):
    type: Literal["ranking"] = "ranking"
    ranking: list[int]
    ranked_message_ids: list[str]
    ranking_parent_id: Optional[str]
    message_tree_id: Optional[str]
    not_rankable: Optional[bool]  # flawed, factually incorrect or unacceptable


class DetoxifyRating(BaseModel):
    toxicity: float
    severe_toxicity: float
    obscene: float
    identity_attack: float
    insult: float
    threat: float
    sexual_explicit: float


class ExportMessageNode(BaseModel):
    message_id: str
    parent_id: str | None
    user_id: str | None
    created_date: datetime | None
    text: str
    role: str
    lang: str | None
    review_count: int | None
    review_result: bool | None
    deleted: bool | None
    rank: int | None
    synthetic: bool | None
    model_name: str | None
    emojis: dict[str, int] | None
    replies: list[ExportMessageNode] | None
    labels: LabelValues | None
    events: dict[str, list[ExportMessageEvent]] | None
    detoxify: DetoxifyRating | None
    # the following fields are always None in message tree exports (see outer tree there)
    message_tree_id: str | None
    tree_state: str | None


class ExportMessageTree(BaseModel):
    message_tree_id: str
    tree_state: Optional[str]
    prompt: Optional[ExportMessageNode]
    origin: Optional[str]

def visit_threads_depth_first(
    node: ExportMessageNode,
    visitor: Callable[[list[ExportMessageNode]], None],
    predicate: Optional[Callable[[list[ExportMessageNode]], bool]] = None,
    parents: list[ExportMessageNode] = None,
):
    parents = parents or []
    if not node:
        return
    thread = parents + [node]
    if predicate is None or predicate(thread):
        visitor(thread)
    if node.replies:
        parents = thread
        for c in node.replies:
            visit_threads_depth_first(node=c, visitor=visitor, predicate=predicate, parents=parents)


def visit_messages_depth_first(
    node: ExportMessageNode,
    visitor: Callable[[ExportMessageNode], None],
    predicate: Optional[Callable[[ExportMessageNode], bool]] = None,
):
    if not node:
        return
    if predicate is None or predicate(node):
        visitor(node)
    if node.replies:
        for c in node.replies:
            visit_messages_depth_first(node=c, visitor=visitor, predicate=predicate)

def open_jsonl_read(input_file_path: str | Path) -> TextIO:
    if not isinstance(input_file_path, Path):
        input_file_path = Path(input_file_path)
    if input_file_path.suffix == ".gz":
        return gzip.open(str(input_file_path), mode="tr", encoding="UTF-8")
    else:
        return input_file_path.open("r", encoding="UTF-8")


def read_oasst_obj(line: str) -> ExportMessageTree | ExportMessageNode:
    dict_tree = json.loads(line)
    # validate data
    if "message_id" in dict_tree:
        return pydantic.parse_obj_as(ExportMessageNode, dict_tree)
    elif "message_tree_id" in dict_tree:
        return pydantic.parse_obj_as(ExportMessageTree, dict_tree)

    raise RuntimeError("Unknown object in jsonl file")


def read_oasst_jsonl(input_file_path: str | Path) -> Iterable[ExportMessageTree | ExportMessageNode]:
    with open_jsonl_read(input_file_path) as file_in:
        # read one object per line
        for line in file_in:
            yield read_oasst_obj(line)


def read_message_trees(input_file_path: str | Path) -> Iterable[ExportMessageTree]:
    for x in read_oasst_jsonl(input_file_path):
        assert isinstance(x, ExportMessageTree)
        yield x


def read_message_tree_list(
    input_file_path: str | Path, filter: Optional[Callable[[ExportMessageTree], bool]] = None
) -> list[ExportMessageTree]:
    return [t for t in read_message_trees(input_file_path) if not filter or filter(t)]


def read_messages(input_file_path: str | Path) -> Iterable[ExportMessageNode]:
    for x in read_oasst_jsonl(input_file_path):
        assert isinstance(x, ExportMessageNode)
        yield x


def read_message_list(
    input_file_path: str | Path, filter: Optional[Callable[[ExportMessageNode], bool]] = None
) -> list[ExportMessageNode]:
    return [t for t in read_messages(input_file_path) if not filter or filter(t)]


class ListDataset(Dataset):
    def __init__(self, data: list):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_oasst_export(
    input_file_path: str | Path,
    val_split: float = 0.2,
    lang: str = "en",
    top_k: Optional[int] = None,
    manual_seed: int = 287631038922,
    data_path: str | Path = None,
    mode: Literal["sft", "rm", "rl"] = "sft",
) -> tuple[ListDataset, ListDataset]:
    if mode not in ("sft", "rm", "rl"):
        raise ValueError(f"Unknown dataset mode: {mode}")

    lang_codes = lang.split(",")

    generator = Generator()
    generator.manual_seed(manual_seed)

    if not isinstance(input_file_path, Path):
        input_file_path = Path(input_file_path)
    if not input_file_path.is_absolute() and data_path:
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        input_file_path = data_path / input_file_path

    threads_per_tree = []
    for tree in read_message_trees(input_file_path):
        if tree.tree_state != "ready_for_export" or not tree.prompt.review_result or tree.prompt.lang not in lang_codes:
            continue

        if mode in ("sft", "rm"):
            if tree.tree_state != "ready_for_export":
                continue
        elif mode == "rl":
            if tree.tree_state not in ("ready_for_export", "prompt_lottery_waiting"):
                continue

        # extract all threads up to last asssitant reply
        threads: list[list[ExportMessageNode]] = []

        def thread_filter(thread: list[ExportMessageNode]) -> bool:
            if any(m.deleted or m.synthetic for m in thread):
                return False

            if top_k is not None:
                for i, m in enumerate(thread):
                    if m.role == "assistant":
                        if m.rank is None:
                            if i > 0 and len(thread[i - 1].replies) > 1:
                                return False
                        elif m.rank >= top_k:
                            return False
            return True

        def leaf_filter(thread: list[ExportMessageNode]) -> bool:
            if mode == "sft":
                # in SFT mode `not thread[-1].replies` finds nodes without children (leaves).
                # We are interested in those which are role='assistant' but some trees don't end on assistant nodes
                # but have prompter leaves .. we want to use those trees too .. e.g. remove the last prompter message(s)
                # so that they end with assistant. The `thread[-2].replies[0] == thread[-1]` check makes sure that only
                # the FIRST prompter reply is added .. e.g. the parent does not appear multiple times and we can use
                # pop() to remove superfluous prompter leaf node later.
                return (
                    len(thread) > 1
                    and not thread[-1].replies
                    and (thread[-1].role == "assistant" or thread[-2].replies[0] == thread[-1])
                    and thread_filter(thread)
                )
            elif mode == "rm":
                # for reward models we use thread-fragments ending on prompter messages as prefix and
                # their (ranked) replies as possible continuations.
                return (
                    thread[-1].role == "prompter"
                    and len([r for r in thread[-1].replies if r.rank is not None]) > 1
                    and thread_filter(thread)
                )
            elif mode == "rl":
                # during rl we are interested in all possible prefixes ending in prompter messages
                return thread[-1].role == "prompter" and not any(m.deleted or m.synthetic for m in thread)

            raise RuntimeError()

        visit_threads_depth_first(tree.prompt, visitor=threads.append, predicate=leaf_filter)
        if mode == "sft":
            for t in threads:
                if t[-1].role == "prompter":
                    t.pop()
        threads_per_tree.append(threads)

    def process_thread(thread):
        if mode == "sft":
            return [m.text for m in thread]
        elif mode == "rm":
            prefix = [m.text for m in thread]
            replies = [r for r in thread[-1].replies if r.role == "assistant" and r.rank is not None]
            replies = sorted(replies, key=lambda r: r.rank)
            replies = [r.text for r in replies]
            return (prefix, replies)
        elif mode == "rl":
            return ([m.text for m in thread],)

        raise RuntimeError()

    # split on tree basis, messages from same tree must not end up in different splits
    trees = ListDataset(threads_per_tree)

    splits = random_split(trees, lengths=[1.0 - val_split, val_split], generator=generator)

    def flatten(ds: ListDataset) -> ListDataset:
        return ListDataset([process_thread(thread) for tree_threads in ds for thread in tree_threads])

    train = flatten(splits[0])
    val = flatten(splits[1])

    print(f"OASST data {str(input_file_path)}: {len(train)=}, {len(val)=}")

    return train, val

if __name__ == '__main__':
    data = load_oasst_export(
        input_file_path='/Users/zhenyishen/Downloads/oasst/2023-04-12_oasst_ready.trees.jsonl.gz',
        lang='en',
        mode= "sft",
    )
    print(data[2])
    print(len(data[2]))