import transformers
import torch
model_name = "jordiclive/falcon_lora_40b_open_assistant"
model_name = "/mnt/data/jordiclive/falcon/save_ckpts/both_sft_ckpt_1400"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)