conda activate jordan_lora

vim /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training/debug_falcon.txt

export PYTHONPATH="/mnt/data/jordiclive/falcon/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/mnt/data/jordiclive/falcon/Open-Assistant/model:$PYTHONPATH"
export HOME="/mnt/data/jordiclive"
export TMP="/mnt/data/jordiclive"
export TEMP="/mnt/data/jordiclive"
export TMPDIR="/mnt/data/jordiclive"
export TRANSFORMERS_CACHE="/mnt/data/jordiclive/transformers_cache"
export HF_DATASETS_CACHE="/mnt/data/jordiclive/transformers_cache"
export HF_HOME="/mnt/data/jordiclive/transformers_cache"
export WANDB_API_KEY="d8216641d549f9bb3d0c5074baa39e15dfd55030"

python /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training/tools/export_lora_model.py --model_path /mnt/data/llama2/Llama-2-70b-hf-sp --ckpt_path /mnt/data/jordiclive/checkpoint-200 --save_merged_model True --output_dir /mnt/data/jordiclive/falcon/ckpt_200  2>&1 | tee debug_falcon.txt
