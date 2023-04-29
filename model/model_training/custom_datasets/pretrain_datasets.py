"""
    Open / close book QA datasets
"""
import glob
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
from datasets import load_dataset
from model_training.custom_datasets.formatting import DatasetEntry, PretrainDatasetEntry
from model_training.custom_datasets.utils import _filter_by_words
from torch import Generator
from torch.utils.data import Dataset, Subset, random_split

# @agoryuno contributed this
re_reference_remove = re.compile(r"\[\d+(?:,\s*\d+)*?\]")
re_single_reference_remove = re.compile(r"\[\s?\d+\s?\]")

# check if the whole string is just a combination of (multiple) whitespaces and newlines
re_whitespace_newline_match = re.compile(r"^[\s\n]*$")


LINKING_CHARS = ["\n", "\n\n", " "]


def index_squad_v2(example):
    if len(example["answers"]["text"]):
        answer = example["answers"]["text"][0]
    else:
        answer = "I do not have answer for that"
    return example["context"] + " " + example["question"], answer


def index_uasquad(example):
    if len(example["Answer"]):
        answer = example["Answer"]
    else:
        answer = "Я не маю на це відповіді"
    return example["Context"] + " " + example["Question"], answer


def index_trivia_qa_nocontext(example):
    # dummy return one randomly
    return example["question"], example["answer"]["aliases"][np.random.randint(len(example["answer"]["aliases"]))]


def index_trivia_qa_context(example):
    question = example["question"]
    if len(example["search_results"]["search_context"]):
        context = example["search_results"]["search_context"][
            np.random.randint(len(example["search_results"]["search_context"]))
        ]
    else:
        context = ""
    answer = example["answer"]["aliases"][np.random.randint(len(example["answer"]["aliases"]))]

    return context + " " + question, answer


def index_adversarial_qa(example):
    return example["title"] + ". " + example["context"] + " " + example["question"], example["answers"]["text"][0]


def index_gsm8k(example):
    return example["question"], example["answer"]


def index_wikihow(example):
    return example["title"] + ", explain step by step", example["result"]


def index_essay_instruction(example):
    return example["instructions"], example["titles"].strip() + "\n" + example["essays"]


def index_math_qa(example):
    """
    we are not including choices, so no need to output the "answer : <a,b,c,d>" part
    > if girls is 10 and boys is 20 , then 10 / 20 . so ratio of girls to boys is = 10 / 20 = 1 / 2 answer : a
    """
    return example["Problem"], example["Rationale"].split("answer : ", maxsplit=1)[0]


def index_eli5(example):
    return example["title"], example["answers"]["text"][0]


def index_gsm_hard(example):
    return example[
        "input"
    ] + "\nWrite a small snippet of python code to answer this", "Here's the code solution to the question\n```python\n{}\n```\n The answer should be {}".format(
        example["code"].strip(), example["target"]
    )





class RedPajama(Dataset):
    name = "red_pajama"

    def __init__(self,  cache_dir: str | Path, mode: str = "sft", char_max_len: str = 9216 ) -> None:
        super().__init__()
        self.mode = mode
        assert mode in ("sft", "rm", "rl")
        self.char_max_len = char_max_len

        self.dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample",cache_dir=cache_dir)['train']


    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> DatasetEntry:
        dialogue = PretrainDatasetEntry(text=self.dataset[index]['text'][:self.char_max_len])
        return dialogue