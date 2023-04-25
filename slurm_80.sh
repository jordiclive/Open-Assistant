#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g80"
#SBATCH --job-name=OA
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out



source /home/jordan/miniconda3/bin/activate open_assistant
cd /home/jordan/Open-Assistant
export TRANSFORMERS_CACHE=/home/jordan/Open-Assistant/transformers_cache
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="/home/jordan/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/home/jordan/Open-Assistant/model:$PYTHONPATH"

#export MASTER_PORT="12802"
#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
#echo "MASTER_ADDR="$MASTER_ADDR
#echo "$MASTER_ADDR slots=8" > /tmp/loopback_hostfile
#export DEEPSPEED_HOSTFILE="/tmp/loopback_hostfile"
#export ADDR="$(hostname -f):29500"
#export MASTER_PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
cd /home/jordan/Open-Assistant/model/model_training/
#srun deepspeed --hostfile $DEEPSPEED_HOSTFILE /admin/home-jordiclive/Open-Assistant/model/model_training/trainer_sft.py --configs defaults oasst_export_eu gpt-neox --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/output_dir --num_train_epochs 8 --residual_dropout 0.2 --deepspeed --num_train_epochs 12 --gradient_accumulation_steps 1 --use_flash_attention false --residual_dropout 0.0 --learning_rate 4e-6

CUDA_VISIBLE_DEVICES=0,1 deepspeed /home/jordan/Open-Assistant/model/model_training/trainer_sft.py --configs defaults marvin2 --cache_dir /home/jordan/Open-Assistant/data_cache --output_dir /home/jordan/Open-Assistant/peft_models/ --deepspeed --residual_dropout 0.0 --use_flash_attention True --peft_model True
