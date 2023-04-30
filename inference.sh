#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g80"
#SBATCH --job-name=OA
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out



source /admin/home-jordiclive/jordan_no/bin/activate
cd /admin/home-jordiclive/peft_open_assistant/Open-Assistant
export TRANSFORMERS_CACHE=/fsx/home-jordiclive/transformers_cache
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export PYTHONPATH="/admin/home-jordiclive/peft_open_assistant/Open-Assistant/model/model_training:$PYTHONPATH"
#export PYTHONPATH="/admin/home-jordiclive/peft_open_assistant/Open-Assistant/model:$PYTHONPATH"


cd /admin/home-jordiclive/peft_open_assistant/Open-Assistant/model/model_eval/manual



#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus 8 ds-zero-inference.py --name OpenAssistant/llama-65b-sft-v7-2k-steps
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus 8 sampling_report_ds_inference.py --model-name /admin/home-jordiclive/llama/7B --config config/noprefix2.json --prompts data/prompt_lottery_en_250_text.jsonl --report /admin/home-jordiclive/65_report.json --verbose --num-samples 2 --half
