source /mnt/data/jordiclive/falcon_lora/bin/activate

cd /mnt/data/jordiclive/falcon/Open-Assistant
export PYTHONPATH="/mnt/data/jordiclive/falcon_lora/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/mnt/data/jordiclive/falcon_lora/Open-Assistant/model:$PYTHONPATH"
export HOME="/mnt/data/jordiclive"
export TMP="/mnt/data/jordiclive"
export TEMP="/mnt/data/jordiclive"
export TMPDIR="/mnt/data/jordiclive"
export TRANSFORMERS_CACHE="/mnt/data/jordiclive/transformers_cache"
export HF_DATASETS_CACHE="/mnt/data/jordiclive/transformers_cache"
export HF_HOME="/mnt/data/jordiclive/transformers_cache"
cd /mnt/data/jordiclive/falcon/Open-Assistant/model/model_eval/manual

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 61000 sampling_report_ds.py --model-name tiiuae/falcon-40b --config config/noprefix2.json --prompts data/prompt_lottery_en_250_text.jsonl --report /mnt/data/jordiclive/report_test.json --verbose --num-samples 2 --half --peft_model jordiclive/falcon_lora_40b_ckpt_500_oasst_1 --model_hidden_size 8192 --dtype "bf16" 2>&1 | tee debug_sampling.txt