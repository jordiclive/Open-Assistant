from dataclasses import dataclass
import torch
from model_training.models.peft_modeling import peft_model
from pathlib import Path
from model_training.utils import (
    get_tokenizer,get_model
)

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
    cache_dir: str  = ""
    model_name: str = ""
    torch_ckpt_path: str = ""


def save_adapter_model_from_ckpt(save_config: SaveLoraConfig):
    tokenizer = get_tokenizer(save_config)
    model = get_model(save_config, tokenizer)
    model = peft_model(model)
    model.load_state_dict(torch.load(save_config.torch_ckpt_path))
    vocab_size = tokenizer.vocab_size
    num_special_tokens = len(tokenizer.additional_special_tokens)

    new_embs = model.state_dict()['base_model.model.model.embed_tokens.weight'][
               vocab_size:vocab_size + num_special_tokens, :].clone()
    new_embs = new_embs.to(save_config.dtype)
    model.save_pretrained(save_config.adapter_save_path, torch_dtype=save_config.dtype)
    tokenizer.save_pretrained(save_config.adapter_save_path)
    torch.save(new_embs, Path(save_config.adapter_save_path).joinpath("extra_embeddings.pt"))



if __name__ == '__main__':
    save_config = SaveLoraConfig(cache_dir="/fsx/home-jordiclive/data_cache",model_name="/admin/home-jordiclive/llama/7B",torch_ckpt_path="/fsx/home-jordiclive/peft_models/_20230422_1339__admin_home-jordiclive_llama_7B_2048/pytorch_model.bin")
    save_adapter_model_from_ckpt(save_config)
