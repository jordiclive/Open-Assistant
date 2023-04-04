import transformers
from peft import PeftModel
import torch
from transformers import GenerationConfig
from huggingface_hub import hf_hub_download

filename = hf_hub_download("jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b", "extra_embeddings.pt")
embed_weights = torch.load(
    filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

device  = 'cuda'
x = torch.load(
    "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b/extra_embeddings.pt",).to(device).to(
    device) # Add special token embeddings
tokenizer = transformers.AutoTokenizer.from_pretrained("jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b")
model = transformers.AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf",torch_dtype=torch.float16) # Load Base Model
model.resize_token_embeddings(32016) # This model repo also contains several embeddings for special tokens that need to be loaded.
lora_weights = "jordiclive/gpt4all-alpaca-oa-codealpaca-lora-7b"
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float16,
) # Load Lora model
model.base_model.model.model.embed_tokens.weight[32000:, :] = embed_weights.to(model.base_model.model.model.embed_tokens.weight.dtype).to(
    device) # Add special token embeddings
model = model.half().to(device)

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams= 4,

)

def generate(prompt,generation_config=generation_config,max_new_tokens=1024,device='cuda'):
    prompt = f"<|prompter|>{prompt}</s><|assistant|>" # OpenAssistant Prompt Format expected
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print("Text generated:")
    print(output)
    return output

if __name__ == '__main__':
    generate("What is a meme, and what's the history behind this word?")
    generate("What's the Earth total population")
    generate("Write a story about future of AI development")

