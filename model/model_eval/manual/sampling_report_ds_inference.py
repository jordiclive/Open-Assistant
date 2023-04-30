import argparse
import gzip
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, Dict, List

import pydantic
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

QA_SPECIAL_TOKENS = {"Question": "<human>", "Answer": "<bot>", "StartPrefix": "<prefix>", "EndPrefix": "</prefix>"}
QA_SPECIAL_TOKENS_V2_5 = {
    "prompter": "<|prompter|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>",
    "prefix_begin": "<|prefix_begin|>",
    "prefix_end": "<|prefix_end|>",
}


class SamplingConfig(pydantic.BaseModel):
    name: Optional[str]
    generate_args: Dict[str, Any] = {}
    pre_text: Optional[str]
    add_prefix_tokens: Optional[bool] = False

    # for legacy mode
    human_name: Optional[str]
    bot_name: Optional[str]


class Configuration(pydantic.BaseModel):
    default: Optional[SamplingConfig]
    configurations: List[SamplingConfig]


class SamplingResult(pydantic.BaseModel):
    sampling_config: str
    sampling_params: dict
    outputs: List[str]


class PromptResults(pydantic.BaseModel):
    prompt: str
    results: List[SamplingResult]


class SamplingReport(pydantic.BaseModel):
    model_name: str
    date: str
    args: dict
    prompts: List[PromptResults]


def load_jsonl(input_file_path: Union[str, Path]) -> List[Union[dict, str]]:
    if not isinstance(input_file_path, Path):
        input_file_path = Path(input_file_path)

    if input_file_path.suffix == ".gz":
        file_in = gzip.open(str(input_file_path), mode="tr", encoding="UTF-8")
    else:
        file_in = input_file_path.open("r", encoding="UTF-8")

    items = []

    with file_in:
        # read one message tree per line
        for line in file_in:
            obj = json.loads(line)
            items.append(obj)

    return items


def sample(
    prompt: str,
    model,
    tokenizer: PreTrainedTokenizer,
    mode: str,
    sampling_config: SamplingConfig,
    device: torch.DeviceObjType,
    skip_input_tokens: bool,
    max_input_len: Optional[int] = None,
):
    assert sampling_config.name, "'name' must be specified for sampling configuration"
    sc = sampling_config
    prefix = ""
    if sampling_config.pre_text:
        if mode == "v2" and sampling_config.add_prefix_tokens:
            prefix = f"<prefix>{sampling_config.pre_text}</prefix>"
        if mode == "v2_5" and sampling_config.add_prefix_tokens:
            prefix = f"{QA_SPECIAL_TOKENS_V2_5['prefix_begin']}{sampling_config.pre_text}{QA_SPECIAL_TOKENS_V2_5['prefix_end']}"
        else:
            prefix = sampling_config.pre_text

    if mode == "v2":
        input_text = f"{prefix}{QA_SPECIAL_TOKENS['Question']}{prompt}{QA_SPECIAL_TOKENS['Answer']}"
    elif mode == "v2_5":
        input_text = f"{prefix}{QA_SPECIAL_TOKENS_V2_5['prompter']}{prompt}{tokenizer.eos_token}{QA_SPECIAL_TOKENS_V2_5['assistant']}"
    else:
        assert sc.human_name and sc.bot_name, "'human_name' and 'bot_name' parameters must be specified in config "
        input_text = f"{prefix}\n{sc.human_name}: {prompt}\n\n{sc.bot_name}: "

    sampling_params = sampling_config.generate_args
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_input_len,
        pad_to_max_length=False,
        truncation=True,
    ).to(device)
    input_ids = inputs.input_ids

    outputs = model.generate(
        input_ids=input_ids,
        pad_token_id=tokenizer.eos_token_id,
        **sampling_params,
    )
    if skip_input_tokens:
        output_tokens = outputs[0, input_ids.size(1) :]
    else:
        output_tokens = outputs[0]
    return output_tokens, sampling_params


def merge_configs(*configs: tuple[Optional[SamplingConfig]]) -> Optional[SamplingConfig]:
    merged: Union[SamplingConfig, None] = None
    for c in configs:
        if not merged:
            if c:
                merged = c.copy(deep=True)
        else:
            # simple fields
            fields = ["name", "pre_text", "human_name", "bot_name", "add_prefix_tokens"]
            for field_name in fields:
                v = getattr(c, field_name)
                if v:
                    setattr(merged, field_name, v)
            # generate args
            if c.generate_args:
                for k, v in c.generate_args.items():
                    merged.generate_args[k] = v

    return merged


def sample_prompt_continuations(
    prompts: List[str],
    model,
    tokenizer: PreTrainedTokenizer,
    mode: str,
    config: Configuration,
    device: torch.DeviceObjType,
    num_samples: int = 1,
    skip_special_tokens: bool = False,
    skip_input_tokens: bool = False,
    verbose: bool = False,
    max_input_len: Optional[int] = None,
) -> List[PromptResults]:
    prompt_results: List[PromptResults] = []
    for p in tqdm(prompts):
        sampling_results: List[SamplingResult] = []
        for sc in config.configurations:
            outputs = []
            for i in range(num_samples):
                if i > 0 and sc.generate_args.get("do_sample") is False:
                    break  # don't repeat greedy sampling
                output_tokens, sampling_params = sample(
                    p,
                    model=model,
                    tokenizer=tokenizer,
                    mode=mode,
                    sampling_config=merge_configs(config.default, sc),
                    device=device,
                    skip_input_tokens=skip_input_tokens,
                    max_input_len=max_input_len,
                )
                output = tokenizer.decode(
                    output_tokens,
                    truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],  # only used for codegen model
                    skip_special_tokens=skip_special_tokens,
                )

                if verbose:
                    print(f"===[ Config: {sc.name} [{i+1}/{num_samples}] ]===\n")
                    print(f'User: "{p}"')
                    print(f'Assistant: "{output}"\n')
                outputs.append(output)

            sampling_results.append(
                SamplingResult(sampling_config=sc.name, sampling_params=sampling_params, outputs=outputs)
            )

        prompt_results.append(PromptResults(prompt=p, results=sampling_results))
    return prompt_results


def load_configs(path: Path) -> Configuration:
    with path.open() as f:
        json_data = json.load(f)

    return pydantic.parse_obj_as(Configuration, json_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str, help="device to use")
    parser.add_argument("--device-index", default=0, type=int, help="device index")
    parser.add_argument("--model-name", type=str, default="facebook/galactica-125m")
    parser.add_argument(
        "--mode",
        type=str,
        default="legacy",
        help="legacy, v2",
    )
    parser.add_argument(
        "--prompts", type=str, help="jsonl string prompts input file name", default="./data/en_100_text.jsonl.gz"
    )
    parser.add_argument("--report", type=str, help="json sampling report output file name")
    parser.add_argument("--seed", type=int, default="42", help="psoudo random number generator seed")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("-n", type=int, help="number of promtps to use (default: all)")
    parser.add_argument("--num-samples", type=int, default=2, help="number of sampling runs per configuration")
    parser.add_argument("--config", type=str, default="config/default.json", help="configuration file path")
    parser.add_argument("--half", action="store_true", default=False, help="use float16")
    parser.add_argument("--int8", action="store_true", default=False, help="use int8 quantization")
    parser.add_argument("--skip-special-tokens", action="store_true", default=False)
    parser.add_argument("--model-type", type=str, default="CausalLM", help="CausalLM, T5Conditional, LLaMA")
    parser.add_argument("--max-input-len", type=int, help="max token counts for input")
    parser.add_argument("--auth-token", type=str)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--peft_model", type=str, default=None)
    parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
    parser.add_argument("--cpu_offload", action="store_true", help="whether to activate CPU offload")
    parser.add_argument("--nvme_offload_path", help="whether to activate NVME offload and the path on nvme")

    return parser.parse_args()


def main():
    """
    Usage example:
    python sampling_report.py --model-name facebook/galactica-125m --config config/default.json --prompts data/en_100_text.jsonl --report report_file.json -n 10 --verbose

    eval oasst model:
    python sampling_report.py --model-name theblackcat102/pythia-3b-deduped-sft --mode v2 --config config/default.json --prompts data/en_100_text.jsonl -n 2 --verbose
    """

    print("Using pytorch version {}".format(torch.__version__))

    args = parse_args()
    if args.int8 and not torch.cuda.is_available():
        print("Warning: --int8 argument passed but cuda is not available. Ignoring --int8.")
        args.int8 = False

    print("Args:", args)

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_threads)

    device = torch.device(args.device, args.device_index)
    print("Device:", device)

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # load configuration
    config = load_configs(Path(args.config))

    model_name = args.model_name
    print(f"Loading model: {model_name}")

    model_args = {}


    from transformers import AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=args.auth_token)
    skip_input_tokens = True


    print("special_tokens_map:", tokenizer.special_tokens_map)
    print(f"eos_token='{tokenizer.eos_token}', eos_token_id={tokenizer.eos_token_id}")

    print("Tokenizer check:")
    input_text = f"{QA_SPECIAL_TOKENS_V2_5['prompter']}Hi!{tokenizer.eos_token}{QA_SPECIAL_TOKENS_V2_5['assistant']}"
    tr = tokenizer(input_text)
    print(tr)
    decoded = tokenizer.decode(tr.input_ids, skip_special_tokens=False)
    print("decoded:", decoded)

    #todo DS LOAD MODEL
    import gc
    import math
    import os
    import time
    from argparse import ArgumentParser

    import torch
    import torch.distributed as dist

    import deepspeed
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.deepspeed import HfDeepSpeedConfig
    from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()

    def print_rank0(*msg):
        if rank != 0:
            return
        print(*msg)

    config = AutoConfig.from_pretrained(model_name)

    # XXX: can't automatically derive dtype via config's `from_pretrained`
    dtype = torch.bfloat16 if model_name in ["bigscience/bloom",
                                             "bigscience/bigscience-small-testing"] else torch.float16

    model_hidden_size = config.hidden_size
    train_batch_size = 1 * world_size

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": model_hidden_size * model_hidden_size,
            "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
            "stage3_param_persistence_threshold": 0,
        },
        "steps_per_print": 2000,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
    }

    if args.cpu_offload and args.nvme_offload_path:
        raise ValueError("Use one of --cpu_offload or --nvme_offload_path and not both")

    if args.cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(device="cpu", pin_memory=True)

    if args.nvme_offload_path:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=True,
            nvme_path=args.nvme_offload_path,
            buffer_size=4e9,
        )

    dschf = HfDeepSpeedConfig(ds_config)  # this tells from_pretrained to instantiate directly on gpus

    if args.benchmark:
        torch.cuda.empty_cache()
        gc.collect()
        deepspeed.runtime.utils.see_memory_usage("pre-from-pretrained", force=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    if args.benchmark:
        deepspeed.runtime.utils.see_memory_usage("post-from-pretrained", force=True)

    model = model.eval()

    print_rank0(ds_config)

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    prompts = load_jsonl(input_file_path=args.prompts)
    prompts = prompts[:5]
    print(f"prompt count: {len(prompts)}")

    if args.n:
        prompts = prompts[: args.n]

    report = SamplingReport(
        model_name=model_name,
        date=datetime.utcnow().isoformat(),
        args=vars(args),
        prompts=sample_prompt_continuations(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            mode=args.mode,
            config=config,
            device=device,
            num_samples=args.num_samples,
            skip_special_tokens=args.skip_special_tokens,
            skip_input_tokens=skip_input_tokens,
            verbose=args.verbose,
            max_input_len=args.max_input_len,
        ),
    )

    report_filename = args.report
    if not report_filename:
        save_model_name = re.sub(r"[^\w\d-]", "_", model_name)
        config_name = Path(args.config).stem
        date = report.date.split("T")[0]
        report_filename = f"{date}_{save_model_name}_sampling_{config_name}.json"
    print("report_filename", report_filename)

    report_path = Path(report_filename)
    print(f"writing report: {str(report_path)}")
    with report_path.open(mode="wt", encoding="UTF-8") as rf:
        x = report.dict(exclude_none=True)
        json.dump(x, rf, indent=2)


if __name__ == "__main__":
    main()
