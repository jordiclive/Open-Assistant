#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g80"
#SBATCH --job-name=OA
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out



source /fsx/home-jordiclive/miniconda3/bin/activate open_assistant
cd /admin/home-jordiclive/Open-Assistant
export TRANSFORMERS_CACHE=/fsx/home-jordiclive/transformers_cache
export PYTHONPATH="/admin/home-jordiclive/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/admin/home-jordiclive/Open-Assistant/model:$PYTHONPATH"


cd /admin/home-jordiclive/Open-Assistant/model/model_training/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed /admin/home-jordiclive/Open-Assistant/model/model_training/trainer_sft.py --configs defaults my-llama-7b --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/output_dir --deepspeed --residual_dropout 0.0 --learning_rate 4e-6 --use_flash_attention True


