source /p/project/ccstdl/clive1/miniconda3/bin/activate mdel

cd /p/project/ccstdl/clive1/Open-Assistant/model/model_training

export PYTHONPATH="/p/project/ccstdl/clive1/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/p/project/ccstdl/clive1/Open-Assistant/model:$PYTHONPATH"
export HOME="/p/project/ccstdl/clive1/"
export TRANSFORMERS_CACHE="/p/project/ccstdl/clive1/transformers_cache"
export HF_DATASETS_CACHE="/p/project/ccstdl/clive1/transformers_cache"
export HF_HOME="/p/project/ccstdl/clive1/transformers_cache"


deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 61500 /p/project/ccstdl/clive1/Open-Assistant/model/model_training/trainer_sft.py --configs defaults dolly15k pythia-70m-deduped --cache_dir /p/project/ccstdl/clive1/data_cache --output_dir /p/project/ccstdl/clive1/output_dir --deepspeed 2>&1 | tee debug_falcon.txt