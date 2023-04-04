from typing import List
import transformers

import torch
from pathlib import Path

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer


def get_llama_model(
        # model/data params
        model,  # the only required argument
        # training hyperparams
        cutoff_len: int = 256,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ['q_proj','k_proj','v_proj','o_proj'],
        # llm hyperparams
        # wandb params
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    # device_map = "auto"
    #
    # model = LlamaForCausalLM.from_pretrained(
    #     base_model,
    #     # load_in_8bit=True,
    #     torch_dtype=torch.float16,
    #     # device_map=device_map,
    # )

    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    # tokenizer.padding_side = "left"  # Allow batched inference

    # model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    return model


def get_specific_model(
        model_name, seq2seqmodel=False, without_head=False, cache_dir=".cache", quantization=False, **kwargs
):
    # encoder-decoder support for Flan-T5 like models
    # for now, we can use an argument but in the future,
    # we can automate this
    if without_head:
        model = transformers.AutoModel.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    elif seq2seqmodel:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    return model


def get_model(tokenizer,llama_path,dtype, pad_vocab_size_to_multiple_of=16):

    model = get_specific_model(
        llama_path,
        cache_dir="/fsx/home-jordiclive/data_cache",
        quantization=False,
        seq2seqmodel=False,
        without_head=False,
        torch_dtype=dtype,
    )

    n_embs = model.get_input_embeddings().num_embeddings

    if (len(tokenizer) != n_embs or pad_vocab_size_to_multiple_of):
        p = pad_vocab_size_to_multiple_of
        target_size = len(tokenizer) if not p else math.ceil(len(tokenizer) / p) * p
        model.resize_token_embeddings(target_size)
    new_embs =model.get_input_embeddings().num_embeddings -  n_embs


    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in model_parameters])
    print("Number of trainable parameters: {}M".format(int(params / 1e6)))


    return model, n_embs, new_embs

from typing import List, NamedTuple

def get_tokenizer(llama_path,cache_dir="/fsx/home-jordiclive/data_cache") -> transformers.AutoTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_path, cache_dir=cache_dir)

    tokenizer_config = match_tokenizer_name(llama_path)

#     if hasattr(conf, "per_digit_tokens") and conf.per_digit_tokens:
#         tokenizer._tokenizer.pre_processor = pre_tokenizers.Digits(True)

    if tokenizer_config.special_tokens:
        if "GPT-JT" in llama_path:
            tokenizer_config.special_tokens.pad_token = tokenizer.eos_token
        # SpecialTokens : latest in 4.25, 4.26
        tokenizer.add_special_tokens(
            {
                "pad_token": tokenizer_config.special_tokens.pad_token,
                "eos_token": tokenizer_config.special_tokens.eos_token,
                "sep_token": tokenizer_config.special_tokens.sep_token,
            }
        )

    additional_special_tokens = (
        []
        if "additional_special_tokens" not in tokenizer.special_tokens_map
        else tokenizer.special_tokens_map["additional_special_tokens"]
    )
    additional_special_tokens = list(set(additional_special_tokens + list(QA_SPECIAL_TOKENS.values())))

    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    return tokenizer

class SpecialTokens(NamedTuple):
    pad_token: str = ""
    eos_token: str = ""
    sep_token: str = ""
class TokenizerConfig(NamedTuple):
    special_tokens: SpecialTokens = {}
def match_tokenizer_name(model_name: str) -> TokenizerConfig:
    """
    Match a partial model name to a tokenizer configuration
    i.e. model_name `Salesforce/codegen-2B-multi` has config name `codegen`
    """
    tokenizer_config_matches = [config for name, config in TOKENIZER_CONFIGS.items() if name in model_name]
    if not tokenizer_config_matches:
        raise ValueError(f"Cannot find any tokeniser configuration to match {model_name=}")
    elif 1 < len(tokenizer_config_matches):
        raise ValueError(f"Found multiple tokeniser configuration matches for {model_name=}")
    else:
        return tokenizer_config_matches[0]



QA_SPECIAL_TOKENS = {
    "Question": "<|prompter|>",
    "Answer": "<|assistant|>",
    "System": "<|system|>",
    "StartPrefix": "<|prefix_begin|>",
    "EndPrefix": "<|prefix_end|>",
}


def format_system_prefix(prefix, eos_token):
    return "{}{}{}".format(
        QA_SPECIAL_TOKENS["System"],
        prefix,
        eos_token,
    )


def format_pairs(pairs, eos_token, add_initial_reply_token=False):
    conversations = [
        "{}{}{}".format(QA_SPECIAL_TOKENS["Question" if i % 2 == 0 else "Answer"], pairs[i], eos_token)
        for i in range(len(pairs))
    ]
    if add_initial_reply_token:
        conversations.append(QA_SPECIAL_TOKENS["Answer"])
    return conversations


def format_rl_text(pairs):
    # convert question answer pairs to only the prefix prompt for RLHF
    return "{}{}{}".format(QA_SPECIAL_TOKENS["Question"], pairs[0], QA_SPECIAL_TOKENS["Answer"])


def format_reply(text, eos_token):
    return "{}{}{}".format(QA_SPECIAL_TOKENS["Answer"], text, eos_token)







import copy
import math
import random
from distutils.util import strtobool
from pathlib import Path
from typing import List, NamedTuple

import evaluate
import torch
import transformers
import yaml
# from model_training.custom_datasets import get_one_dataset
# from model_training.custom_datasets.formatting import QA_SPECIAL_TOKENS
# from model_training.losses import CrossEntropyLoss, PolyLoss, RMLoss
# from model_training.models import freeze_top_n_layers, get_specific_model
# from model_training.models.patching import patch_model
# from model_training.models.reward_model import GPTNeoXRewardModel
from sklearn.model_selection import train_test_split
from tokenizers import pre_tokenizers
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler


def _strtobool(x):
    return bool(strtobool(x))


class PerDatasetSampler(DistributedSampler):
    """Sampler which returns a fixed number of samples per dataset, per epoch.

    Example:

    Dataset 1 has 10,000 examples and we want 200 per epoch
    Dataset 2 has 500 examples and we want all 500 per epoch

    Epoch size will be 700 and every epoch we'll sample a different
    200 from dataset 1.

    Parameters
    ----------
    dataset_sizes : List[int]
        A list with the size of each dataset.
    dataset_size_per_epoch : List[int]
        How many examples to get from each dataset per epoch.

    Note: dataset_sizes & dataset_size_per_epoch must be in the same order.
    Further the examples in the underlying torch.utils.data.Dataset
    must per ordered as dataset_1, dataset_2, ..., dataset_n. This is fine
    if we concatenate a bunch of datasets together
    e.g. using torch.utils.data.ConcatDataset which is current behaviour.
    """

    def __init__(
        self,
        dataset_sizes: List[int],
        dataset_size_per_epoch: List[int],
        rank: int = None,
        world_size: int = None,
        shuffle: bool = True,
        seed: int = 0,
        samples_length: List[int] = None,
    ):
        """
        if samples_length is not None, then the sampler
        will order the samples by dataset length
        with some variability across epochs
        """
        self.dataset_sizes = dataset_sizes
        self.dataset_size_per_epoch = dataset_size_per_epoch
        self.num_datasets = len(dataset_sizes)
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        if world_size == 1:
            self.rank = 0

        self.num_samples = sum(dataset_size_per_epoch)
        self.seed = seed
        self.samples_length = samples_length

    def set_epoch(self, epoch) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples // self.world_size

    def __iter__(self):
        epoch_idx = []
        n = 0

        random.seed(self.epoch + self.seed)

        for i in range(self.num_datasets):
            sampled_idx = random.sample(range(n, self.dataset_sizes[i] + n), self.dataset_size_per_epoch[i])
            n += self.dataset_sizes[i]
            epoch_idx.extend(sampled_idx)

        if self.samples_length is not None:
            # sort by samples length and in case of ties randomize
            epoch_idx = sorted(epoch_idx, key=lambda x: (self.samples_length[x], random.random()))

            if self.shuffle:
                # do some minor shuffling to avoid repeating the same order
                # but not too much to avoid too much padding
                # quasi random basically
                for i in range(0, len(epoch_idx), 200):  # this should be batch_size dependent
                    random.shuffle(epoch_idx[i : i + 200])
        else:
            if self.shuffle:
                random.shuffle(epoch_idx)

        # split epoch_idx in world_size chunks
        epoch_idx = epoch_idx[self.rank : self.num_samples : self.world_size]

        return iter(epoch_idx)

    @classmethod
    def build_sampler_from_config(cls, training_conf, datasets, *args, **kwargs):
        dataset_sizes = [len(x) for x in datasets]
        fractions = get_dataset_fractions(training_conf.datasets, dataset_sizes, verbose=training_conf.verbose)
        dataset_size_per_epoch = [int(size * frac) for size, frac in zip(dataset_sizes, fractions)]
        return cls(dataset_sizes, dataset_size_per_epoch, *args, **kwargs)


def get_dataset_fractions(conf, dataset_sizes, verbose=False):
    """Calculate fraction of each dataset to use per epoch when subsampling"""

    if verbose:
        print("Creating sampler for datasets:")

    fractions = []
    for i, data_config in enumerate(conf):
        dataset_name, _ = get_dataset_name_and_kwargs_from_data_config(data_config)
        if isinstance(data_config, dict):
            if "fraction" in data_config[dataset_name]:
                if data_config[dataset_name]["fraction"] <= 0:
                    raise ValueError("Please specify fraction as a value between 0 < fraction <= 1")
                fractions.append(min(1, data_config[dataset_name]["fraction"]))
            elif "size" in data_config[dataset_name]:
                if data_config[dataset_name]["size"] > dataset_sizes[i]:
                    raise ValueError(f"Please specify a size smaller than number of examples: {dataset_sizes[i]:,.0f}")
                fractions.append(data_config[dataset_name]["size"] / dataset_sizes[i])
            else:
                fractions.append(1)
        else:
            fractions.append(1)

        if verbose:
            print(f"Dataset: {dataset_name} fraction chosen: {fractions[-1]:.2f}")
    return fractions


class SpecialTokens(NamedTuple):
    pad_token: str = ""
    eos_token: str = ""
    sep_token: str = ""


class TokenizerConfig(NamedTuple):
    special_tokens: SpecialTokens = {}


TOKENIZER_CONFIGS = {
    "galactica": TokenizerConfig(special_tokens=SpecialTokens("<pad>", "</s>")),
    "GPT-JT": TokenizerConfig(special_tokens=SpecialTokens(sep_token="<|extratoken_100|>")),
    "codegen": TokenizerConfig(special_tokens=SpecialTokens("<|endoftext|>", sep_token="<|endoftext|>")),
    "pythia": TokenizerConfig(special_tokens=SpecialTokens("<|padding|>", "<|endoftext|>", "<|endoftext|>")),
    "gpt-neox": TokenizerConfig(special_tokens=SpecialTokens("<|padding|>", "<|endoftext|>", "<|endoftext|>")),
    "llama": TokenizerConfig(special_tokens=SpecialTokens("</s>", "</s>", sep_token="<s>")),
}


def save_adapter(torch_path,llama_path,adapter_save_path,dtype=torch.float16):
    tokenizer = get_tokenizer("/admin/home-jordiclive/llama/7B")
    model, n_embs, new_embs = get_model(tokenizer,llama_path,dtype=dtype)
    model = get_llama_model(model)
    model.load_state_dict(torch.load(torch_path))
    new_embs = model.state_dict()['base_model.model.model.embed_tokens.weight'][n_embs:,:]
    model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    torch.save(new_embs,  Path(adapter_save_path).joinpath("extra_embeddings.pt"))


save_adapter(torch_path="/fsx/home-jordiclive/output_dir_20230404_204017_decapoda-research/llama-13b-hf_2048/checkpoint-20/pytorch_model.bin",llama_path="decapoda-research/llama-13b-hf",adapter_save_path="/fsx/home-jordiclive/adapter",dtype=torch.float16)

import os

import torch
import transformers
from peft import PeftModel


if torch.cuda.is_available():
    device = "cuda"



def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "/fsx/home-jordiclive/adapter",
):
    tokenizer = transformers.AutoTokenizer.from_pretrained("/fsx/home-jordiclive/adapter")
    if device == "cuda":
        model, n_embs, new_embs = get_model(tokenizer,"/admin/home-jordiclive/llama/7B",dtype=torch.float16)
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
        p = 16
        target_size = len(tokenizer) if not p else math.ceil(len(tokenizer) / p) * p
        model.resize_token_embeddings(32016)
        model.base_model.model.model.embed_tokens.weight[32000:, :] = torch.load("/fsx/home-jordiclive/adapter/extra_embeddings.pt").to(model.base_model.model.model.embed_tokens.weight.dtype).to(device)

        model = model.half().to("cuda")
        while True:
            text = input("\n\nInput text to prompt the model: ")
            text = str(text)
            if len(text) == 0:
                continue
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            from transformers import GenerationConfig
            temperature = 0.1
            top_p = 0.75
            top_k = 40
            num_beams = 4
            max_new_tokens = 128
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,

            )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            print("Text generated:")
            print(output)

            # # add the length of the prompt tokens to match with the mesh-tf generation
            # max_length = 400 + ids.shape[1]
            #
            # gen_tokens = model.generate(
            #     ids,
            #     do_sample=True,
            #     min_length=max_length,
            #     max_length=max_length,
            #     temperature=0.9,
            #     use_cache=True
            # )
            # gen_text = tokenizer.batch_decode(gen_tokens)[0]
            print("Text generated:")
            # print(gen_text)




main()
