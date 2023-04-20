from peft import PeftModel
import torch
from huggingface_hub import hf_hub_download
from transformers import GenerationConfig


class OAPeftModel(PeftModel):
    def generate(self,input_ids,pad_token_id,**kwargs):
        generation_config = GenerationConfig(**kwargs)
        output = self.generate(input_ids=input_ids,pad_token_id=pad_token_id,eos_token_id=self.eos_token_id,
        generation_config=generation_config)
        return output


            #no_repeat_ngram_size=3,

        # input_ids,
        # ** sampling_params,
        # pad_token_id = tokenizer.eos_token_id,



def load_peft_model(model, peft_model_path,tokenizer):
    model.resize_token_embeddings(
        len(tokenizer)
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model = OAPeftModel.from_pretrained(
        model,
        peft_model_path,
        torch_dtype=model.dtype,
    )
    model.eos_token_id = tokenizer.eos_token_id
    extra_embeds = hf_hub_download(peft_model_path, "extra_embeddings.pt")
    embed_weights = torch.load(
        extra_embeds, map_location=model.device
    )
    model.base_model.model.model.embed_tokens.weight[len(tokenizer) - embed_weights.shape[0]:, :] = embed_weights.to(
        model.base_model.model.model.embed_tokens.weight.dtype
    )
    return model