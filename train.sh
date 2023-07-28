conda activate jordan_lora

cd /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training

export PYTHONPATH="/mnt/data/jordiclive/falcon/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/mnt/data/jordiclive/falcon/Open-Assistant/model:$PYTHONPATH"
export HOME="/mnt/data/jordiclive"
export TMP="/mnt/data/jordiclive"
export TEMP="/mnt/data/jordiclive"
export TMPDIR="/mnt/data/jordiclive"
export TRANSFORMERS_CACHE="/mnt/data/jordiclive/transformers_cache"
export HF_DATASETS_CACHE="/mnt/data/jordiclive/data_cache"
export HF_HOME="/mnt/data/jordiclive/transformers_cache"
export WANDB_API_KEY="d8216641d549f9bb3d0c5074baa39e15dfd55030"

deepspeed --include=localhost:0,1,2,3,4,5,6,7 /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training/trainer_sft.py --cache_dir /mnt/data/jordiclive/data_cache --output_dir /mnt/data/jordiclive/falcon/orca_2 --configs llama2_13b_orcacode2_8k --show_dataset_stats --deepspeed 2>&1 | tee debug_falcon.txt