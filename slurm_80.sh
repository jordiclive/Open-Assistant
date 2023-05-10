#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g80"
#SBATCH --job-name=bash
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out



source /admin/home-jordiclive/miniconda3/bin/activate open_assistant
cd /admin/home-jordiclive/peft_open_assistant/Open-Assistant
export TRANSFORMERS_CACHE=/fsx/home-jordiclive/transformers_cache
export PYTHONPATH="/admin/home-jordiclive/peft_open_assistant/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/admin/home-jordiclive/peft_open_assistant/Open-Assistant/model:$PYTHONPATH"

#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
cd /admin/home-jordiclive/peft_open_assistant/Open-Assistant/model/model_training/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed /admin/home-jordiclive/peft_open_assistant/Open-Assistant/model/model_training/trainer_sft.py --configs defaults lora-finetune lora-llama-30b --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/peft_models_lora_30b_normal/ --deepspeed --residual_dropout 0.0 --wandb_name "30B_continue" --r_value 16 --resume_from_checkpoint_pth "/fsx/home-jordiclive/peft_models_lora_30b/_20230508_0700__admin_home-jordiclive_llama_7B_2048/checkpoint-12000" 2>&1 | tee debug_30.txt

##srun deepspeed --hostfile $DEEPSPEED_HOSTFILE /admin/home-jordiclive/Open-Assistant/model/model_training/trainer_sft.py --configs defaults oasst_export_eu gpt-neox --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/output_dir --num_train_epochs 8 --residual_dropout 0.2 --deepspeed --num_train_epochs 12 --gradient_accumulation_steps 1 --use_flash_attention false --residual_dropout 0.0 --learning_rate 4e-6
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed /admin/home-jordiclive/peft_open_assistant/Open-Assistant/model/model_training/trainer_sft.py --configs defaults lora-finetune lora-llama-65b --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/peft_models_lora_65B_r_2/ --deepspeed --residual_dropout 0.0 2>&1 --wandb_name "65B continue"| tee debug_65.txt
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed /admin/home-jordiclive/peft_open_assistant/Open-Assistant/model/model_training/trainer_sft.py --configs defaults lora-finetune lora-llama-30b --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/peft_models_lora_30b_2/ --deepspeed --residual_dropout 0.0 --wandb_name "30B_continue" 2>&1 | tee debug_30.txt
#
#
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed /admin/home-jordiclive/peft_open_assistant/Open-Assistant/model/model_training/trainer_sft.py --configs defaults lora-finetune lora-llama-30b --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/peft_models_lora_30b_normal/ --deepspeed --residual_dropout 0.0 --wandb_name "30B_continue" --r_value 16 --resume_from_checkpoint_pth "/fsx/home-jordiclive/peft_models_lora_30b_2/_20230508_0740__admin_home-jordiclive_llama_7B_2048/checkpoint-6000" 2>&1 | tee debug_30.txt
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed /admin/home-jordiclive/peft_open_assistant/Open-Assistant/model/model_training/trainer_sft.py --configs defaults lora-finetune lora-llama-30b --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/peft_models_lora_30b_r_64/ --deepspeed --residual_dropout 0.0 --wandb_name "30B_r_64_continue" --r_value 64 --resume_from_checkpoint_pth "/fsx/home-jordiclive/peft_models_lora_30b/_20230508_0700__admin_home-jordiclive_llama_7B_2048/checkpoint-12000" 2>&1 | tee debug_30_r.txt

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed /admin/home-jordiclive/peft_open_assistant/Open-Assistant/model/model
#_training/trainer_sft.py --configs defaults pythia-12B --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/peft_models_lora_13b_4/ --deepspeed --residual_dropout 0.0


# r = 16 /fsx/home-jordiclive/peft_models_lora_30b_2/_20230508_0740__admin_home-jordiclive_llama_7B_2048/checkpoint-6000
#r = 64 /fsx/home-jordiclive/peft_models_lora_30b/_20230508_0700__admin_home-jordiclive_llama_7B_2048/checkpoint-12000