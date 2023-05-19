

source /mnt/data/jordiclive/jordan/bin/activate
cd /mnt/data/jordiclive/Open-Assistant
export TRANSFORMERS_CACHE=/mnt/data/jordiclive/transformers_cache
export PYTHONPATH="/mnt/data/jordiclive/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/mnt/data/jordiclive/Open-Assistant/model:$PYTHONPATH"

#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
cd /mnt/data/jordiclive/Open-Assistant/model/model_training
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 deepspeed /mnt/data/jordiclive/Open-Assistant/model/model_training/trainer_sft.py --configs defaults lora-finetune lora-llama-65b --cache_dir /home/ubuntu/data_cache --output_dir /mnt/data/jordiclive/65B_ckpts --deepspeed --residual_dropout 0.0 --r_value 16 2>&1 | tee debug_30.txt