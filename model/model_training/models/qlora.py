# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
from datasets import load_dataset
import evaluate
import nltk

from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"





def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

#
# class SavePeftModelCallback(transformers.TrainerCallback):
#     def save_model(self, args, state, kwargs):
#         print('Saving PEFT checkpoint...')
#         if state.best_model_checkpoint is not None:
#             checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
#         else:
#             checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
#
#         peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
#         kwargs["model"].save_pretrained(peft_model_path)
#
#         pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
#         if os.path.exists(pytorch_model_path):
#             os.remove(pytorch_model_path)
#
#     def on_save(self, args, state, control, **kwargs):
#         self.save_model(args, state, kwargs)
#         return control
#
#     def on_train_end(self, args, state, control, **kwargs):
#         def touch(fname, times=None):
#             with open(fname, 'a'):
#                 os.utime(fname, times)
#
#         touch(join(args.output_dir, 'completed'))
#         self.save_model(args, state, kwargs)
#
#
def get_accelerate_model(args, checkpoint_dir):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}

    if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map='auto',
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type  # {'fp4', 'nf4'}
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('=' * 80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('=' * 80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    modules = find_all_linear_names(args, model)

    model.config.torch_dtype = (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    if not args.full_finetune:
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'))
            for name, p in model.named_parameters():
                if 'lora' in name:
                    print(name, p.sum())
        else:
            print(f'adding LoRA modules...')
            model = get_peft_model(model, config)

    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model
#
#
# def print_trainable_parameters(args, model):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     if args.bits == 4: trainable_params /= 2
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param}")
#
#
# def smart_tokenizer_and_embedding_resize(
#         special_tokens_dict: Dict,
#         tokenizer: transformers.PreTrainedTokenizer,
#         model: transformers.PreTrainedModel,
# ):
#     """Resize tokenizer and embedding.
#
#     Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
#     """
#     num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
#     model.resize_token_embeddings(len(tokenizer))
#
#     if num_new_tokens > 0:
#         input_embeddings = model.get_input_embeddings().weight.data
#         output_embeddings = model.get_output_embeddings().weight.data
#
#         input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
#         output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
#
#         input_embeddings[-num_new_tokens:] = input_embeddings_avg
#         output_embeddings[-num_new_tokens:] = output_embeddings_avg
#
#
# @dataclass
# class DataCollatorForCausalLM(object):
#     tokenizer: transformers.PreTrainedTokenizer
#     source_max_len: int
#     target_max_len: int
#     train_on_source: bool
#     predict_with_generate: bool
#
#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         # Extract elements
#         sources = [example['input'] for example in instances]
#         targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
#         # Tokenize
#         tokenized_sources_with_prompt = self.tokenizer(
#             sources,
#             max_length=self.source_max_len,
#             truncation=True,
#         )
#         tokenized_targets = self.tokenizer(
#             targets,
#             max_length=self.target_max_len,
#             truncation=True,
#             add_special_tokens=False,
#         )
#         # Build the input and labels for causal LM
#         input_ids = []
#         labels = []
#         for tokenized_source, tokenized_target in zip(
#                 tokenized_sources_with_prompt['input_ids'],
#                 tokenized_targets['input_ids']
#         ):
#             if not self.predict_with_generate:
#                 input_ids.append(torch.tensor(tokenized_source + tokenized_target))
#                 if not self.train_on_source:
#                     labels.append(
#                         torch.tensor(
#                             [IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
#                     )
#                 else:
#                     labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
#             else:
#                 input_ids.append(torch.tensor(tokenized_source))
#         # Apply padding
#         input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
#         labels = pad_sequence(labels, batch_first=True,
#                               padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
#         data_dict = {
#             'input_ids': input_ids,
#             'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
#         }
#         if labels is not None:
#             data_dict['labels'] = labels
#         return data_dict
#
#
# def extract_unnatural_instructions_data(examples, extract_reformulations=False):
#     out = {
#         'input': [],
#         'output': [],
#     }
#     for example_instances in examples['instances']:
#         for instance in example_instances:
#             out['input'].append(instance['instruction_with_input'])
#             out['output'].append(instance['output'])
#     if extract_reformulations:
#         for example_reformulations in examples['reformulations']:
#             if example_reformulations is not None:
#                 for instance in example_reformulations:
#                     out['input'].append(instance['instruction_with_input'])
#                     out['output'].append(instance['output'])
#     return out
#
#
#
#
def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training

import argparse
from enum import Enum
from transformers import GenerationConfig

class IntervalStrategy(Enum):
    NO = 'no'
    STEPS = 'steps'

class SchedulerType(Enum):
    CONSTANT = 'constant'

class OptimizerNames(Enum):
    ADAMW_HF = 'adamw_hf'

class HubStrategy(Enum):
    EVERY_SAVE = 'every_save'

# Create argparse Namespace object


def get_qlora_model():
    import argparse
    from enum import Enum
    from transformers import GenerationConfig

    class IntervalStrategy(Enum):
        NO = 'no'
        STEPS = 'steps'

    class SchedulerType(Enum):
        CONSTANT = 'constant'

    class OptimizerNames(Enum):
        ADAMW_HF = 'adamw_hf'

    class HubStrategy(Enum):
        EVERY_SAVE = 'every_save'

    # Create argparse Namespace object
    args = argparse.Namespace(
        model_name_or_path='/mnt/data/llama_hf/7B',
        trust_remote_code=False,
        eval_dataset_size=1024,
        max_train_samples=None,
        max_eval_samples=None,
        source_max_len=1024,
        target_max_len=256,
        dataset='alpaca',
        output_dir='./output',
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        do_predict=False,
        evaluation_strategy=IntervalStrategy.NO,
        prediction_loss_only=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=8,
        per_gpu_train_batch_size=None,
        per_gpu_eval_batch_size=None,
        gradient_accumulation_steps=16,
        eval_accumulation_steps=None,
        eval_delay=0,
        learning_rate=0.0002,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        max_grad_norm=0.3,
        num_train_epochs=3.0,
        max_steps=10000,
        lr_scheduler_type=SchedulerType.CONSTANT,
        warmup_ratio=0.03,
        warmup_steps=0,
        log_level='passive',
        log_level_replica='warning',
        log_on_each_node=True,
        logging_dir='./output/runs/May27_19-10-42_Jordans-MBP.broadband',
        logging_strategy=IntervalStrategy.STEPS,
        logging_first_step=False,
        logging_steps=10,
        logging_nan_inf_filter=True,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=250,
        save_total_limit=40,
        save_on_each_node=False,
        no_cuda=False,
        use_mps_device=False,
        seed=42,
        data_seed=None,
        jit_mode_eval=False,
        use_ipex=False,
        bf16=False,
        fp16=False,
        fp16_opt_level='O1',
        half_precision_backend='auto',
        bf16_full_eval=False,
        fp16_full_eval=False,
        tf32=None,
        local_rank=-1,
        xpu_backend=None,
        tpu_num_cores=None,
        tpu_metrics_debug=False,
        debug=[],
        dataloader_drop_last=False,
        eval_steps=None,
        dataloader_num_workers=0,
        past_index=-1,
        run_name='./output',
        disable_tqdm=False,
        remove_unused_columns=False,
        label_names=None,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
        ignore_data_skip=False,
        sharded_ddp=[],
        fsdp=[],
        fsdp_min_num_params=0,
        fsdp_config={'fsdp_min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},
        fsdp_transformer_layer_cls_to_wrap=None,
        deepspeed=None,
        label_smoothing_factor=0.0,
        optim=OptimizerNames.ADAMW_HF,
        optim_args=None,
        adafactor=False,
        group_by_length=True,
        length_column_name='length',
        report_to=[],
        ddp_find_unused_parameters=None,
        ddp_bucket_cap_mb=None,
        dataloader_pin_memory=True,
        skip_memory_metrics=True,
        use_legacy_prediction_loop=False,
        push_to_hub=False,
        resume_from_checkpoint=None,
        hub_model_id=None,
        hub_strategy=HubStrategy.EVERY_SAVE,
        hub_token=None,
        hub_private_repo=False,
        gradient_checkpointing=True,
        include_inputs_for_metrics=False,
        fp16_backend='auto',
        push_to_hub_model_id=None,
        push_to_hub_organization=None,
        push_to_hub_token=None,
        mp_parameters='',
        auto_find_batch_size=False,
        full_determinism=False,
        torchdynamo=None,
        ray_scope='last',
        ddp_timeout=1800,
        torch_compile=False,
        torch_compile_backend=None,
        torch_compile_mode=None,
        sortish_sampler=False,
        predict_with_generate=False,
        generation_max_length=None,
        generation_num_beams=None,
        generation_config=GenerationConfig(
            max_new_tokens=256,
            transformers_version="4.28.0.dev0"
        ),
        cache_dir=None,
        train_on_source=False,
        mmlu_split='eval',
        mmlu_dataset='mmlu-fs',
        do_mmlu_eval=False,
        max_mmlu_samples=None,
        mmlu_source_max_len=2048,
        full_finetune=False,
        adam8bit=False,
        double_quant=True,
        quant_type='nf4',
        bits=4,
        lora_r=64,
        lora_alpha=16,
        lora_dropout=0.0,
        max_memory_MB=80000,
        _n_gpu=0,
        __cached__setup_devices='device(type="cpu")'
    )
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model = get_accelerate_model(args, checkpoint_dir)
    return model

if __name__ == '__main__':
    get_qlora_model()