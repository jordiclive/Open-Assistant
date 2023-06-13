source /mnt/data/jordiclive/falcon_lora/bin/activate

cd /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training

export PYTHONPATH="/mnt/data/jordiclive/falcon_lora/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/mnt/data/jordiclive/falcon_lora/Open-Assistant/model:$PYTHONPATH"
export HOME="/mnt/data/jordiclive"
export TMP="/mnt/data/jordiclive"
export TEMP="/mnt/data/jordiclive"
export TMPDIR="/mnt/data/jordiclive"
export TRANSFORMERS_CACHE="/mnt/data/jordiclive/transformers_cache"
export HF_DATASETS_CACHE="/mnt/data/jordiclive/transformers_cache"
export HF_HOME="/mnt/data/jordiclive/transformers_cache"
cd /mnt/data/jordiclive/falcon/Open-Assistant/model/model_training/models

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 61500 test_model.py 