from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "tiiuae/falcon-40b"
repo_id = "jordiclive/falcon_lora_40b_open_assistant"
dtype = torch.bfloat16


tokenizer = AutoTokenizer.from_pretrained(repo_id)


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


def load_peft_finetuned_model(model, peft_model_path, tokenizer):
    peft_model_path = Path(peft_model_path).joinpath("adapter")
    embed_weights = hf_hub_download(peft_model_path, "extra_embeddings.pt")
    adapter_model = hf_hub_download(peft_model_path, "adapter_model.bin")
    add_embeddings(model, embed_weights, tokenizer)
    adapters_weights = torch.load(adapter_model, map_location=model.device)
    model.load_state_dict(adapters_weights, strict=False)
    model.eos_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def load_lora_model(base_model_id, repo_id, tokenizer, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    peft_model_path = Path(repo_id).joinpath("adapter")
    config_path = hf_hub_download(peft_model_path, "adapter_config.json")

    config = LoraConfig.from_pretrained(config_path)
    model = get_peft_model(model, config)
    model = load_peft_finetuned_model(model, repo_id, tokenizer)
    return model


model = load_lora_model(base_model_id=base_model_id, repo_id=repo_id, tokenizer=tokenizer,  dtype=dtype)
print(model)
