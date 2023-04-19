from transformers import AutoModel
import torch
import tempfile
import os

def shard_model(model_path,new_path):
    model = AutoModel.from_pretrained(model_path,torch_dtype=torch.float16)
    model.save_pretrained(new_path, max_shard_size="1024MB")

if __name__ == "__main__":
    shard_model("/fsx/home-jordiclive/66B_checkpoints/output_dir_20230417_1816_decapoda-research/llama-65b-hf_2048/checkpoint-3000","/fsx/home-jordiclive/sharded_66B_ckpt")
