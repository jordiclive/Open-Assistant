source /mnt/data/jordiclive/falcon_lora/bin/activate

cd /mnt/data/jordiclive/falcon/Open-Assistant
export TRANSFORMERS_CACHE=/mnt/data/jordiclive/transformers_cache
export PYTHONPATH="/mnt/data/jordiclive/falcon_lora/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/mnt/data/jordiclive/falcon_lora/Open-Assistant/model:$PYTHONPATH"
#export PYTHONPATH="/mnt/data/jordiclive/Open-Assistant/model/model_training:$PYTHONPATH"
#export PYTHONPATH="/mnt/data/jordiclive/Open-Assistant/model:$PYTHONPATH"

#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
cd /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training/trainer_sft.py --configs defaults oasst-top1 falcon-40b --cache_dir /mnt/data/jordiclive/data_cache --output_dir /mnt/data/jordiclive/falcon 2>&1 --deepspeed | tee debug_30.txt
#export CUDA_VISIBLE_DEVICES=0,1
#deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 61000 /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training/trainer_sft.py --configs defaults oasst-top1 falcon-40b --cache_dir /mnt/data/jordiclive/data_cache --output_dir /mnt/data/jordiclive/falcon 2>&1 --deepspeed | tee debug_30.txt
deepspeed --include=localhost:0,1 --master_port 61500 /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training/trainer_sft.py --configs defaults oasst-top1 falcon-7b-lora --cache_dir /mnt/data/jordiclive/data_cache --output_dir /mnt/data/jordiclive/falcon 2>&1 --deepspeed | tee debug_falcon.txt


#CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES="0,1" /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training/trainer_sft.py --configs defaults sft-8-datasets falcon-7b --cache_dir /mnt/data/jordiclive/data_cache --output_dir /mnt/data/jordiclive/falcon --residual_dropout 0.0 2>&1 | tee debug_30.txt
