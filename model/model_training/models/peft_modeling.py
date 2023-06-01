from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from model_training.utils.utils import get_model, get_tokenizer
from peft import LoraConfig, PeftModel, PrefixTuningConfig, get_peft_model, prepare_model_for_int8_training
from peft.tuners.lora import LoraLayer


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


# def find_all_linear_names(args, model):
#     cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, cls):
#             names = name.split('.')
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])
#
#
#     if 'lm_head' in lora_module_names: # needed for 16-bit
#         lora_module_names.remove('lm_head')
#     return list(lora_module_names)

# def get_peft_model(model,peft_type='lora',int8_training=False, gradient_checkpointing=False):
#     falcon= ['dense_4h_to_h', 'dense', 'query_key_value', 'dense_h_to_4h']
#     llama = ['down_proj', 'k_proj', 'q_proj', 'gate_proj', 'o_proj', 'up_proj', 'v_proj']
#     # modules = find_all_linear_names(args, model)
#     modules = falcon
#     config = LoraConfig(
#         r=args.lora_r,
#         lora_alpha=args.lora_alpha,
#         target_modules=modules,
#         lora_dropout=args.lora_dropout,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )


def peft_model(model, peft_type="lora", int8_training=False, gradient_checkpointing=False, bf16=True):
    if peft_type == "lora":
        config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["dense_4h_to_h", "dense", "query_key_value", "dense_h_to_4h"],
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

    # for name, module in model.named_modules():
    #     if isinstance(module, LoraLayer):
    #         if bf16:
    #             module = module.to(torch.bfloat16)
    #     if 'norm' in name:
    #         module = module.to(torch.float32)
    #     if 'lm_head' in name or 'embed_tokens' in name:
    #         if hasattr(module, 'weight'):
    #             if args.bf16 and module.weight.dtype == torch.float32:
    #                 module = module.to(torch.bfloat16)

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
        torch_ckpt_path="/mnt/data/jordiclive/falcon/checkpoint-500/pytorch_model.bin",
        adapter_save_path="/mnt/data/jordiclive/falcon_lora_checkpoint_500",
        model_name="tiiuae/falcon-40b",
        cache_dir="/mnt/data/jordiclive/data_cache",
        dtype=torch.bfloat16,
        peft_type="lora",
        residual_dropout_lima=0.3,
    )
    save_adapter_model_from_ckpt(save_config)