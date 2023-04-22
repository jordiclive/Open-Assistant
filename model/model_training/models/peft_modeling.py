from typing import Callable

from peft import PeftModel
import torch
from huggingface_hub import hf_hub_download
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    PrefixTuningConfig
)

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

def peft_model(model,peft_type='lora',int8_training=False):

    if peft_type=='lora':
        config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    elif peft_type=='prefix-tuning':
        config = PrefixTuningConfig(num_virtual_tokens=30,prefix_projection=True,encoder_hidden_size=1024)
    else:
        raise ValueError('peft_method config is lora or prefix-tuning')
    model = get_peft_model(model, config)
    if int8_training:
        model = prepare_model_for_int8_training(model)
    model.print_trainable_parameters()
    return model


from transformers import AdamW
import math
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

class CustomAdamW(AdamW):
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if 'lora' or 'prefix' not in p.grad_fn.variable.name:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


class PeftFlashTrainer(SFTTrainer):
    def create_optimizer(self):
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = CustomAdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer