from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from model_training.utils.utils import get_model, get_tokenizer
from peft import LoraConfig, PeftModel, PrefixTuningConfig, get_peft_model, prepare_model_for_int8_training


def load_peft_model(model, peft_model_path, tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(
        model,
        peft_model_path,
        torch_dtype=model.dtype,
    )
    model.eos_token_id = tokenizer.eos_token_id
    extra_embeds = hf_hub_download(peft_model_path, "extra_embeddings.pt")
    embed_weights = torch.load(extra_embeds, map_location=model.device)
    model.base_model.model.model.embed_tokens.weight[len(tokenizer) - embed_weights.shape[0] :, :] = embed_weights.to(
        model.base_model.model.model.embed_tokens.weight.dtype
    )
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


def peft_model(model, peft_type="lora", int8_training=False, gradient_checkpointing=False):
    if peft_type == "lora":
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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

def load_peft_ckpt(model, tokenizer,peft_ckpt_path=None):
    model = PeftModel.from_pretrained(
        model,
        model_id="/mnt/data/jordiclive/adapter_ckpt_10500",
        torch_dtype=torch.float16,
    )
    # model.eos_token_id = tokenizer.eos_token_id
    #
    #
    # embed_weights = torch.load("/mnt/data/jordiclive/adapter_ckpt_10500/extra_embeddings.pt",map_location=model.device)
    # print('embed_requires_grad1',embed_weights.requires_grad)
    # embed_weights.requires_grad = False
    #
    # model.base_model.model.model.embed_tokens.weight.data[32000:32000+embed_weights.shape[0], :] = embed_weights.data.to(
    # model.base_model.model.model.embed_tokens.weight.dtype
    # ).to(
    # model.base_model.model.model.embed_tokens.weight.device
    # )
    # print('embed_requires_grad',model.base_model.model.model.embed_tokens.weight.requires_grad)
    model = prepare_model_for_gradient_checkpointing(model)
    model.print_trainable_parameters()
    return model

def transfer_embeddings(model,path):
    from transformers.deepspeed import  is_deepspeed_zero3_enabled

    old_embeddings = model.get_input_embeddings()
    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    else:
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_embeddings = torch.nn.Embedding(old_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
    model._init_weights(new_embeddings)
    embed_weights = torch.load(path,map_location=old_embeddings.weight.device)
    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                new_embeddings.weight.data[:32000, :] = old_embeddings.weight.data[:32000, :]
                new_embeddings.weight.data[32000:32000+embed_weights.shape[0], :] = embed_weights.weight.data.to(new_embeddings.weight.dtype).to(new_embeddings.weight.device)
    else:
        new_embeddings.weight.data[:32000, :] = old_embeddings.weight.data[:32000, :]
        new_embeddings.weight.data[32000:32000 + embed_weights.shape[0], :] = embed_weights.weight.data.to(
            new_embeddings.weight.dtype).to(new_embeddings.weight.device)

    model.set_input_embeddings(new_embeddings)
    model.tie_weights()

def transfer_embeddings_after(model,path):
    from transformers.deepspeed import  is_deepspeed_zero3_enabled

    old_embeddings = model.get_input_embeddings()
    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    else:
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_embeddings = torch.nn.Embedding(old_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
    model._init_weights(new_embeddings)
    embed_weights = torch.load(path,map_location=old_embeddings.weight.device)
    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                new_embeddings.weight.data[:32000, :] = old_embeddings.weight.data[:32000, :]
                new_embeddings.weight.data[32000:32000+embed_weights.shape[0], :] = embed_weights.weight.data.to(new_embeddings.weight.dtype).to(new_embeddings.weight.device)
    else:
        new_embeddings.weight.data[:32000, :] = old_embeddings.weight.data[:32000, :]
        new_embeddings.weight.data[32000:32000 + embed_weights.shape[0], :] = embed_weights.weight.data.to(
            new_embeddings.weight.dtype).to(new_embeddings.weight.device)

    model.set_input_embeddings(new_embeddings)
    model.tie_weights()




@dataclass
class SaveLoraConfig:
    dtype: torch.dtype = torch.float16
    is_reward_model: bool = False
    quantization: bool = False
    seq2seqmodel: bool = False
    freeze_layer: bool = False
    residual_dropout: float = 0
    use_flash_attention: bool = False
    adapter_save_path: str = "adapter"
    cache_dir: str = ""
    model_name: str = ""
    torch_ckpt_path: str = ""
    peft_type: str = "lora"


def save_adapter_model_from_ckpt(save_config: SaveLoraConfig):
    tokenizer = get_tokenizer(save_config)
    model = get_model(save_config, tokenizer)
    model = peft_model(model)
    model.load_state_dict(torch.load(save_config.torch_ckpt_path))
    vocab_size = tokenizer.vocab_size
    num_special_tokens = len(tokenizer.additional_special_tokens)

    new_embs = model.state_dict()["base_model.model.model.embed_tokens.weight"][
        vocab_size : vocab_size + num_special_tokens, :
    ].clone()
    new_embs = new_embs.to(save_config.dtype)
    model.save_pretrained(save_config.adapter_save_path, torch_dtype=save_config.dtype)
    tokenizer.save_pretrained(save_config.adapter_save_path)
    torch.save(new_embs, Path(save_config.adapter_save_path).joinpath("extra_embeddings.pt"))

if __name__ == '__main__':
    save_config = SaveLoraConfig(
        torch_ckpt_path="/mnt/data/jordiclive/65B_ckpts_stage2/checkpoint-666/pytorch_model.bin",
        adapter_save_path="/mnt/data/jordiclive/stage_2_oasst_top_1_adapter_ckpt_666",
        model_name="/mnt/data/llama_hf/65B",
        cache_dir="/mnt/data/jordiclive/data_cache",
        dtype=torch.float16,
        peft_type="lora",
    )
    save_adapter_model_from_ckpt(save_config)