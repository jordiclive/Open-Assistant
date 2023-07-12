import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, PeftModel, PrefixTuningConfig, get_peft_model, prepare_model_for_int8_training
import transformers
from tokenizers import pre_tokenizers

QA_SPECIAL_TOKENS = {
    "Question": "<|prompter|>",
    "Answer": "<|assistant|>",
    "System": "<|system|>",
    "StartPrefix": "<|prefix_begin|>",
    "EndPrefix": "<|prefix_end|>",
}

class SpecialTokens(NamedTuple):
    pad_token: str = ""
    eos_token: str = ""
    sep_token: str = ""

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
        if "falcon" in model_name:
            kwargs["trust_remote_code"] = True
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    return model


class TokenizerConfig(NamedTuple):
    special_tokens: SpecialTokens = {}
TOKENIZER_CONFIGS = {
    "galactica": TokenizerConfig(special_tokens=SpecialTokens("<pad>", "</s>")),
    "GPT-JT": TokenizerConfig(special_tokens=SpecialTokens(sep_token="<|extratoken_100|>")),
    "codegen": TokenizerConfig(special_tokens=SpecialTokens("<|endoftext|>", sep_token="<|endoftext|>")),
    "pythia": TokenizerConfig(special_tokens=SpecialTokens("<|padding|>", "<|endoftext|>", "<|endoftext|>")),
    "gpt-neox": TokenizerConfig(special_tokens=SpecialTokens("<|padding|>", "<|endoftext|>", "<|endoftext|>")),
    "llama": TokenizerConfig(special_tokens=SpecialTokens("</s>", "</s>", sep_token="<s>")),
    "cerebras": TokenizerConfig(special_tokens=SpecialTokens("<|endoftext|>", "<|endoftext|>", "<|endoftext|>")),
    "deberta-v3": TokenizerConfig(special_tokens=SpecialTokens("[PAD]", "[SEP]", sep_token="[CLS]")),
    "bloom": TokenizerConfig(special_tokens=SpecialTokens("<pad>", "</s>", "<s>")),
    "electra": TokenizerConfig(special_tokens=SpecialTokens("[PAD]", "[SEP]", sep_token="[CLS]")),
    "falcon": TokenizerConfig(
        special_tokens=SpecialTokens("<|endoftext|>", "<|endoftext|>", sep_token="<|endoftext|>")
    ),
}

def match_tokenizer_name(model_name: str):
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

def get_tokenizer(conf) -> transformers.AutoTokenizer:
    tokenizer_name = conf.model_name

    if "cerebras" in conf.model_name:
        # Only 13B has a tokenizer available on HF
        tokenizer_name = "cerebras/Cerebras-GPT-13B"

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=conf.cache_dir)

    tokenizer_config = match_tokenizer_name(conf.model_name)

    if hasattr(conf, "per_digit_tokens") and conf.per_digit_tokens:
        tokenizer._tokenizer.pre_processor = pre_tokenizers.Digits(True)

    if tokenizer_config.special_tokens:
        if "GPT-JT" in conf.model_name:
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



def get_model(conf, tokenizer, pad_vocab_size_to_multiple_of=16, check_freeze_layer=True):
    dtype = torch.float32
    if conf.dtype in ["fp16", "float16"]:
        dtype = torch.float16
    elif conf.dtype in ["bf16", "bfloat16"]:
        dtype = torch.bfloat16


    if not conf.is_reward_model:

        model = get_specific_model(
            conf.model_name,
            cache_dir=conf.cache_dir,
            quantization=conf.quantization,
            seq2seqmodel=conf.seq2seqmodel,
            without_head=conf.is_reward_model,
            torch_dtype=dtype,
        )

        n_embs = model.get_input_embeddings().num_embeddings
        if len(tokenizer) != n_embs and check_freeze_layer:
            assert not conf.freeze_layer, "Cannot change the number of embeddings if the model is frozen."

        if len(tokenizer) != n_embs or pad_vocab_size_to_multiple_of:
            p = pad_vocab_size_to_multiple_of
            target_size = len(tokenizer) if not p else math.ceil(len(tokenizer) / p) * p
            print("Resizing embeddings to", target_size)
            model.resize_token_embeddings(target_size)



    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in model_parameters])
    print("Number of trainable parameters: {}M".format(int(params / 1e6)))

    # patch_model(
    #     model,
    #     resid_pdrop=conf.residual_dropout,
    #     flash_attention=conf.use_flash_attention,
    #     residual_dropout_lima=conf.residual_dropout_lima,
    # )

    return model


def add_embeddings(model, embed_path, tokenizer):
    old_embeddings = model.get_input_embeddings()
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_embeddings = torch.nn.Embedding(old_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
    model._init_weights(new_embeddings)
    embed_weights = torch.load(embed_path, map_location=old_embeddings.weight.device)
    vocab_size = tokenizer.vocab_size
    new_embeddings.weight.data[:vocab_size, :] = old_embeddings.weight.data[:vocab_size, :]
    new_embeddings.weight.data[vocab_size : vocab_size + embed_weights.shape[0], :] = embed_weights.to(
        new_embeddings.weight.dtype
    ).to(new_embeddings.weight.device)
    model.set_input_embeddings(new_embeddings)
    model.tie_weights()


def load_peft_model(model, peft_model_path, tokenizer):
    embed_weights = hf_hub_download(peft_model_path, "extra_embeddings.pt")
    model.resize_token_embeddings(tokenizer.vocab_size + torch.load(embed_weights).shape[0])
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(
        model,
        model_id=peft_model_path,
        torch_dtype=model.dtype,
    )
    model.eos_token_id = tokenizer.eos_token_id
    add_embeddings(model, embed_weights, tokenizer)
    return model


def load_peft_finetuned_model(model, peft_model_path, tokenizer):
    add_embeddings(model, Path(peft_model_path).joinpath("extra_embeddings.pt"), tokenizer)
    adapters_weights = torch.load(Path(peft_model_path).joinpath("adapter_model.bin"), map_location=model.device)
    model.load_state_dict(adapters_weights, strict=False)
    return model


def prepare_model_for_gradient_checkpointing(model):
    r"""
    Prepares the model for gradient checkpointing if necessary
    """
    if not getattr(model, "is_loaded_in_8bit", False):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model


def peft_model(model, model_name, peft_type="lora", int8_training=False, gradient_checkpointing=False):
    if peft_type == "lora":
        if "falcon" in model_name:
            target_modules = ["dense_4h_to_h", "dense", "query_key_value", "dense_h_to_4h"]
            r = 64
        elif "llama" in model_name:
            target_modules = ["down_proj", "k_proj", "q_proj", "gate_proj", "o_proj", "up_proj", "v_proj"]
            if "65" in model_name:
                r = 16
            else:
                r = 64
        else:
            raise ValueError(
                f"Invalid model name '{model_name}'. The model name should contain 'falcon' or 'llama', Simply find target_modules for it"
            )
        config = LoraConfig(
            r=r,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif peft_type == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=30, prefix_projection=True, encoder_hidden_size=1024, task_type="CAUSAL_LM"
        )
    else:
        raise ValueError("peft_method config is lora or prefix-tuning")
    model = get_peft_model(model, config)
    if int8_training:
        model = prepare_model_for_int8_training(model)

    if gradient_checkpointing:
        model = prepare_model_for_gradient_checkpointing(model)
    model.print_trainable_parameters()
    return model


@dataclass
class SaveLoraConfig:
    dtype: torch.dtype = torch.float16
    is_reward_model: bool = False
    quantization: bool = False
    seq2seqmodel: bool = False
    freeze_layer: bool = False
    residual_dropout: float = 0
    residual_dropout_lima: float = 0.3
    use_flash_attention: bool = False
    adapter_save_path: str = "adapter"
    cache_dir: str = ""
    model_name: str = ""
    torch_ckpt_path: str = ""
    peft_type: str = "lora"


def save_adapter_model_from_ckpt(save_config: SaveLoraConfig):
    tokenizer = get_tokenizer(save_config)
    model = get_model(save_config, tokenizer)
    model = peft_model(model, model_name=save_config.model_name)
    model.load_state_dict(torch.load(save_config.torch_ckpt_path))
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size is {vocab_size}, and new tokenizer length is {len(tokenizer)}")
    old_embeddings = model.get_input_embeddings()
    new_embs = old_embeddings.weight.data[vocab_size:, :].clone()
    new_embs = new_embs.to(save_config.dtype)
    model.save_pretrained(save_config.adapter_save_path, torch_dtype=save_config.dtype)
    torch.save(new_embs, Path(save_config.adapter_save_path).joinpath("extra_embeddings.pt"))
    tokenizer.save_pretrained(save_config.adapter_save_path)



def save_merged_model_from_ckpt(save_config: SaveLoraConfig):
    tokenizer = get_tokenizer(save_config)
    model = get_model(save_config, tokenizer)
    model = load_peft_model(model, save_config.adapter_save_path, tokenizer)
    model = model.merge_and_unload()
    model = model.to(save_config.dtype)
    model.save_pretrained(Path(save_config.adapter_save_path).joinpath("merged_model"), dtype=save_config.dtype, max_shard_size="10GB")
    tokenizer.save_pretrained(save_config)
    # model = peft_model(model, model_name=save_config.model_name)
    # model = load_peft_model(model, peft_model_path, tokenizer)
    # model.load_state_dict(torch.load(save_config.torch_ckpt_path))
    # model = model.merge_and_unload()
    # # model = model.to(save_config.dtype)
    # model.save_pretrained(save_config,dtype=save_config.dtype, max_shard_size="10GB")
    # tokenizer.save_pretrained(save_config)


def load_peft_model_merge(model, peft_model_path, tokenizer):
    embed_weights = peft_model_path.joinpath("extra_embeddings.pt")
    model.resize_token_embeddings(tokenizer.vocab_size + torch.load(embed_weights).shape[0])
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(
        model,
        model_id=peft_model_path,
        torch_dtype=model.dtype,
    )
    model.eos_token_id = tokenizer.eos_token_id
    add_embeddings(model, embed_weights, tokenizer)
    return model



def save_merged_model_again(save_config):
    tokenizer = get_tokenizer(save_config)
    model = get_model(save_config, tokenizer)
    model = peft_model(model, save_config.model_name, peft_type="lora", int8_training=False, gradient_checkpointing=True)
    model = load_peft_finetuned_model(model, peft_model_path= "/mnt/data/jordiclive/falcon/falcon-lora-1.1k", tokenizer=tokenizer)

    model = model.merge_and_unload()
    model = model.to(save_config.dtype)  # todo needed?

    model.save_pretrained("merged_falcon2", dtype=save_config.dtype, max_shard_size="10GB")
    tokenizer.save_pretrained("merged_falcon2")

if __name__ == '__main__':
    save_config = SaveLoraConfig(dtype=torch.bfloat16,
                                 model_name="tiiuae/falcon-40b",
                                 cache_dir="/mnt/data/jordiclive/data_cache",
                                 adapter_save_path="",
                                 torch_ckpt_path="")
    save_merged_model_again(save_config)


