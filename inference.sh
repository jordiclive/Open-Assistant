#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g80"
#SBATCH --job-name=OA
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out



source /admin/home-jordiclive/jordan_no/bin/activate
cd /admin/home-jordiclive/peft_open_assistant/Open-Assistant
export TRANSFORMERS_CACHE=/fsx/home-jordiclive/transformers_cache
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export PYTHONPATH="/admin/home-jordiclive/peft_open_assistant/Open-Assistant/model/model_training:$PYTHONPATH"
#export PYTHONPATH="/admin/home-jordiclive/peft_open_assistant/Open-Assistant/model:$PYTHONPATH"


cd /admin/home-jordiclive/peft_open_assistant/Open-Assistant/Open-Assistant/model/model_training/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus 8 ds-zero-inference.py --name OpenAssistant/llama-65b-sft-v7-2k-steps