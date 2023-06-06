from peft import prepare_model_for_int8_training


def load_model(
    base_model,
    base_model_config,
    model_type,
    tokenizer,
    cfg,
    adapter="lora",
    inference=False,
):
    # type: (str, str, str, str, DictDefault, Optional[str], bool) -> Tuple[PreTrainedModel, Optional[PeftConfig]]
    """
    Load a model from a base model and a model type.
    """

    # TODO refactor as a kwarg
    load_in_8bit = cfg.load_in_8bit
    is_llama_derived_model = "llama" in base_model or (
        cfg.model_type and "llama" in cfg.model_type.lower()
    )
    if cfg.bf16:
        torch_dtype = torch.bfloat16
    elif cfg.load_in_8bit or cfg.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32



    model_kwargs = {}
    if cfg.adapter == "qlora" and cfg.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    try:
        if cfg.gptq and is_llama_derived_model:
            from alpaca_lora_4bit.autograd_4bit import load_llama_model_4bit_low_ram
            from huggingface_hub import snapshot_download

            try:
                snapshot_download_kwargs = {}
                if cfg.base_model_ignore_patterns:
                    snapshot_download_kwargs[
                        "ignore_patterns"
                    ] = cfg.base_model_ignore_patterns
                cache_model_path = Path(
                    snapshot_download(base_model, **snapshot_download_kwargs)
                )
                files = (
                    list(cache_model_path.glob("*.pt"))
                    + list(cache_model_path.glob("*.safetensors"))
                    + list(cache_model_path.glob("*.bin"))
                )
                if len(files) > 0:
                    model_path = str(files[0])
                else:
                    logging.warning(
                        "unable to find a cached model file, this will likely fail..."
                    )
                    model_path = str(cache_model_path)
            except Exception:  # pylint: disable=broad-exception-caught
                model_path = cfg.base_model
            model, _ = load_llama_model_4bit_low_ram(
                base_model_config if base_model_config else base_model,
                model_path,
                device_map=cfg.device_map,
                half=cfg.fp16,
                groupsize=cfg.gptq_groupsize if cfg.gptq_groupsize else -1,
                is_v1_model=cfg.gptq_model_v1
                if cfg.gptq_model_v1 is not None
                else True,
            )
            load_in_8bit = False
        elif is_llama_derived_model and "LlamaForCausalLM" in globals():
            config = LlamaConfig.from_pretrained(base_model_config)
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                config=config,
                load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                torch_dtype=torch_dtype,
                device_map="auto" if cfg.world_size == 1 else cfg.device_map,
                **model_kwargs,
            )
        # elif model_type == "GPTNeoXForCausalLM" and cfg.flash_attention:
        #     This is a WIP, still an issue with the backward pass
        #     RuntimeError: grad can be implicitly created only for scalar outputs
        #     TODO: try config.sequence_parallel = False
        #     # https://github.com/HazyResearch/flash-attention/blob/40a25c8ee7465cf547b929cfa2937034e37bfce9/tests/models/test_gpt_neox.py#L12
        #     # https://github.com/HazyResearch/flash-attention/tree/main/training#model-components
        #     # add `**kwargs` to https://github.com/HazyResearch/flash-attention/blob/40a25c8ee7465cf547b929cfa2937034e37bfce9/flash_attn/models/gpt.py#L442
        #     from flash_attn.utils.pretrained import state_dict_from_pretrained
        #     from flash_attn.models.gpt import GPTLMHeadModel
        #     from flash_attn.models.gpt_neox import remap_state_dict_hf_gpt_neox, gpt_neox_config_to_gpt2_config
        #     from transformers import GPTNeoXConfig
        #     config = gpt_neox_config_to_gpt2_config(GPTNeoXConfig.from_pretrained(base_model))
        #     config.use_flash_attn = True
        #     config.fused_bias_fc = True
        #     config.fused_mlp = True  # GPT-NeoX-20B uses "gelu_fast"
        #     config.activation_function = "gelu_fast"
        #     config.fused_dropout_add_ln = True
        #     # config.residual_in_fp32 = True
        #
        #     model: GPTLMHeadModel = GPTLMHeadModel.from_pretrained(
        #         base_model,
        #         config,
        #         dtype=torch_dtype,
        #         device=cfg.device,
        #     )
        #     model.train() # sets to train instead of eval mode
        elif model_type:
            model = getattr(transformers, model_type).from_pretrained(
                base_model,
                load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                torch_dtype=torch_dtype,
                device_map=cfg.device_map,
                trust_remote_code=cfg.trust_remote_code or False,
                **model_kwargs,
            )
        else:
            config = AutoConfig.from_pretrained(
                base_model,
                trust_remote_code=cfg.trust_remote_code or False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                config=config,
                load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
                load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
                torch_dtype=torch_dtype,
                device_map=cfg.device_map,
                trust_remote_code=cfg.trust_remote_code or False,
                **model_kwargs,
            )
    except Exception as err:  # pylint: disable=broad-exception-caught
        logging.error(
            "Exception raised attempting to load model, retrying with AutoModelForCausalLM"
        )
        logging.exception(err)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
            torch_dtype=torch_dtype,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code or False,
            **model_kwargs,
        )

    embeddings_len = math.ceil(len(tokenizer) / 32) * 32
    model.resize_token_embeddings(embeddings_len)

    if not cfg.gptq and (
        (cfg.adapter == "lora" and load_in_8bit)
        or (cfg.adapter == "qlora" and cfg.load_in_4bit)
    ):
        logging.info("converting PEFT model w/ prepare_model_for_int8_training")
        model = prepare_model_for_int8_training(model)

    model, lora_config = load_adapter(model, cfg, adapter)

    if cfg.ddp and not load_in_8bit:
        model.to(f"cuda:{cfg.local_rank}")

    if cfg.gptq:
        # Scales to half
        logging.info("Fitting 4bit scales and zeros to half")
        for _, module in model.named_modules():
            if "Autograd4bitQuantLinear" in str(type(module)) or "Linear4bitLt" in str(
                type(module)
            ):
                if hasattr(module, "is_v1_model") and module.is_v1_model:
                    module.zeros = module.zeros.half()
                module.scales = module.scales.half()
                module.bias = module.bias.half()

    if (
        torch.cuda.device_count() > 1
        and int(os.getenv("WORLD_SIZE", "1")) > 1
        and (cfg.gptq or cfg.load_in_4bit)
    ):
        # llama is PROBABLY model parallelizable, but the default isn't that it is
        # so let's only set it for the 4bit, see
        # https://github.com/johnsmith0031/alpaca_lora_4bit/blob/08b3fca4a4a9e0d3945be1bab4529f100a428636/finetune.py#L130-L133
        setattr(model, "is_parallelizable", True)
        setattr(model, "model_parallel", True)

    requires_grad = []
    for name, param in model.named_parameters(recurse=True):
        if param.requires_grad:
            requires_grad.append(f"{name}: {param.requires_grad}")
    if len(requires_grad) == 0:
        logging.warning("there are no parameters that require gradient updates")
    model.config.use_cache = False

    # TODO resume_from_checkpoint handling
    return model, lora_config