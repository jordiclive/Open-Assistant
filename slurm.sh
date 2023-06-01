source /mnt/data/jordiclive/falcon_lora/bin/activate

cd /mnt/data/jordiclive/falcon/Open-Assistant
export TRANSFORMERS_CACHE=/mnt/data/jordiclive/transformers_cache
export PYTHONPATH="/mnt/data/jordiclive/falcon_lora/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/mnt/data/jordiclive/falcon_lora/Open-Assistant/model:$PYTHONPATH"


cd /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training
export WANDB_API_KEY = "d8216641d549f9bb3d0c5074baa39e15dfd55030"
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 61500 /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training/trainer_sft.py --configs defaults oasst-top1 falcon-40b-lora --cache_dir /mnt/data/jordiclive/data_cache --output_dir /mnt/data/jordiclive/falcon --deepspeed 2>&1 | tee debug_falcon.txt