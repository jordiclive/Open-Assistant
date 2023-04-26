from dataclasses import dataclass
import torch
from model_training.models.peft_modeling import peft_model
from pathlib import Path
import argparse
@dataclass
class SaveLoraConfig:
    cache_dir: str = "/fsx/home-jordiclive/data_cache"
    model_name: str = "/admin/home-jordiclive/llama/7B"
    dtype: torch.dtype = torch.float16
    is_reward_model: bool = False
    quantization: bool = False
    seq2seqmodel: bool = False
    freeze_layer: bool = False
    residual_dropout: float = 0
    use_flash_attention: bool = False
    torch_ckpt_path: str =  "/fsx/home-jordiclive/peft_models/_20230423_1715__admin_home-jordiclive_llama_7B_2048/checkpoint-1407/pytorch_model.bin"
    adapter_save_path: str = "adapter"
training_conf = SaveLoraConfig()
from model_training.utils import (
    get_tokenizer,get_model
)
tokenizer = get_tokenizer(training_conf)
model = get_model(training_conf, tokenizer)
model = peft_model(model)

import argparse


# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Process two torch model paths")
#
#     parser.add_argument("--torch_path1", type=str, required=True,
#                         help="Path to the first torch model file")
#
#     parser.add_argument("--torch_path2", type=str, required=True,
#                         help="Path to the second torch model file")
#
#     args = parser.parse_args()
#
#     return args


if __name__ == "__main__":
    # args = parse_arguments()
    print('BASE MODE',model.state_dict()['base_model.model.model.layers.16.mlp.gate_proj.weight'])
    print('BASE MODEL_LORA',model.state_dict()['base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight'])
    x = model.state_dict()['base_model.model.model.layers.16.mlp.gate_proj.weight'].clone()
    y = model.state_dict()['base_model.model.model.embed_tokens.weight'].clone()

    torch_path1 = "/fsx/home-jordiclive/peft_models/_20230426_1626__admin_home-jordiclive_llama_7B_2048/checkpoint-100/pytorch_model.bin"
    torch_path2 = "/fsx/home-jordiclive/peft_models/_20230426_1626__admin_home-jordiclive_llama_7B_2048/checkpoint-200/pytorch_model.bin"

    model.load_state_dict(torch.load(torch_path1, map_location=torch.device('cpu')))
    vocab_size = tokenizer.vocab_size
    print('CKpt1',model.state_dict()['base_model.model.model.layers.16.mlp.gate_proj.weight'])
    print('Ckpt1_LORA',model.state_dict()['base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight'])


    model.load_state_dict(torch.load(torch_path2, map_location=torch.device('cpu')))
    print('CKpt2',model.state_dict()['base_model.model.model.layers.16.mlp.gate_proj.weight'])
    print('Ckpt2_LORA',model.state_dict()['base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight'])



# model.load_state_dict(torch.load(training_conf.torch_ckpt_path))
# vocab_size = tokenizer.vocab_size
# num_special_tokens = len(tokenizer.additional_special_tokens)

# new_embs = model.state_dict()['base_model.model.model.embed_tokens.weight'][vocab_size:vocab_size+num_special_tokens,:].clone()
# new_embs = new_embs.to(training_conf.dtype)
# model.save_pretrained(training_conf.adapter_save_path,torch_dtype=training_conf.dtype)
# tokenizer.save_pretrained(training_conf.adapter_save_path)
# torch.save(new_embs,  Path(training_conf.adapter_save_path).joinpath("extra_embeddings.pt"))