source /mnt/data/jordiclive/falcon_lora/bin/activate

cd /mnt/data/jordiclive/falcon/Open-Assistant/model/model_eval/manual

export PYTHONPATH="/mnt/data/jordiclive/falcon_lora/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/mnt/data/jordiclive/falcon_lora/Open-Assistant/model:$PYTHONPATH"
export HOME="/mnt/data/jordiclive"
export TMP="/mnt/data/jordiclive"
export TEMP="/mnt/data/jordiclive"
export TMPDIR="/mnt/data/jordiclive"
export TRANSFORMERS_CACHE="/mnt/data/jordiclive/transformers_cache"
export HF_DATASETS_CACHE="/mnt/data/jordiclive/transformers_cache"
export HF_HOME="/mnt/data/jordiclive/transformers_cache"


deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 61500 sampling_report_ds2.py --model-name OpenAssistant/falcon-40b-lora-sft-1.1k --config config/noprefix2.json --prompts data/prompt_lottery_en_250_text.jsonl --verbose --mode v2_5 --half  2>&1 | tee debug_falcon.txt