from pathlib import Path

import torch
import transformers
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import GenerationConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
repo_id = "jordiclive/falcon_lora_40b_ckpt_500_oasst_1"
base_model = "tiiuae/falcon-40b"

dtype = torch.float16
repo_id = "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b"
base_model = "decapoda-research/llama-7b-hf"
# Model Loading
def add_embeddings(model, embed_path, tokenizer):
    old_embeddings = model.get_input_embeddings()
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_embeddings = torch.nn.Embedding(old_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
    model._init_weights(new_embeddings)
    embed_weights = torch.load(embed_path, map_location=old_embeddings.weight.device)
    vocab_size = tokenizer.vocab_size
    new_embeddings.weight.data[:vocab_size, :] = old_embeddings.weight.data[:vocab_size, :]
    new_embeddings.weight.data[vocab_size : vocab_size + embed_weights.shape[0], :] = embed_weights.weight.data.to(
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


tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id)

model = transformers.AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=dtype, trust_remote_code=True, cache_dir="/mnt/data/jordiclive/data_cache"
)
model = load_peft_model(model, repo_id, tokenizer)


# device  configuration
model = model.to(device)


# Choose Generation parameters

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
)


def format_system_prompt(prompt, eos_token="</s>"):
    return "{}{}{}{}".format("<|prompter|>", prompt, eos_token, "<|assistant|>")


def generate(prompt, generation_config=generation_config, max_new_tokens=2048, device=device):
    prompt = format_system_prompt(prompt)  # OpenAssistant Prompt Format expected
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            eos_token_id=model.eos_token_id,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print("Text generated:")
    print(output)
    return output


generate("What is a meme, and what's the history behind this word?")
generate("What's the Earth total population")
generate("Write a story about future of AI development")
