conda activate ikkaenv

cd /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training

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

python peft_merge_no_deps.py 2>&1 | tee export.txt