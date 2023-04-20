from peft import PeftModel
import torch
from huggingface_hub import hf_hub_download


def load_peft_model(model, peft_model_path,tokenizer):
    model.resize_token_embeddings(
        len(tokenizer)
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(
        model,
        peft_model_path,
        torch_dtype=torch.float16,
    )
    model.eos_token_id = tokenizer.eos_token_id
    extra_embeds = hf_hub_download(peft_model_path, "extra_embeddings.pt")
    embed_weights = torch.load(
        extra_embeds, map_location=model.device
    )
    model.base_model.model.model.embed_tokens.weight[len(tokenizer) - embed_weights:, :] = embed_weights.to(
        model.base_model.model.model.embed_tokens.weight.dtype
    )
    return model