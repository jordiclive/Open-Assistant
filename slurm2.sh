#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g40"
#SBATCH --job-name=flan
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out
source ~/miniconda3/bin/activate meta_conda
cd /fsx/home-jordiclive/metadata




export MODEL=gpt2-xl
export NUM_GPU=8
export DEEPSPEED_CONFIG=$(realpath bsmetadata/deepspeed_configs/v2.json)
export DATA_DIR=$(realpath local-data)
echo "deepspeed_config_file: $DEEPSPEED_CONFIG"


echo "compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: $DEEPSPEED_CONFIG
distributed_type: DEEPSPEED
fp16: true
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
num_machines: 1
num_processes: -1
mixed_precision: fp16
" > accelerate_config.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file accelerate_config.yaml bsmetadata/train.py --config-name v2 \
  model_name=$MODEL \
    data_config.train_file='*.jsonl.gz' \
    data_config.validation_file='c4-en-html_cc-main-2019-18_pq00-001.jsonl.gz' \
    data_config.dataset_name=$DATA_DIR \
    data_config.preprocessing_num_workers=6  extra_steps_to_eval_save_at='[2,100,200,400,800]' \
    data_config.metadata_config.metadata_list='[html]' \
    data_config.metadata_config.metadata_column_list='[html]' \
    out_dir=/fsx/home-jordiclive/tmp/metadata-run-html