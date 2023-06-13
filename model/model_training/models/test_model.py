import transformers
model_name = "jordiclive/falcon_lora_40b_open_assistant"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)