#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g40"
#SBATCH --job-name=OA
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out



source /fsx/home-jordiclive/miniconda3/bin/activate open
cd /admin/home-jordiclive/Open-Assistant
export TRANSFORMERS_CACHE=/fsx/home-jordiclive/transformers_cache
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="/fsx/home-jordiclive/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/fsx/home-jordiclive/Open-Assistant/model:$PYTHONPATH"

cd /admin/home-jordiclive/Open-Assistant/model/model_training/models

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python peft_modeling.py