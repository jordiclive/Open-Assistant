#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g40"
#SBATCH --job-name=OA
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out

module load openmpi
module load cuda/11.7

source /fsx/home-jordiclive/miniconda3/bin/activate open

cd /admin/home-jordiclive/Open-Assistant
export TRANSFORMERS_CACHE=/fsx/home-jordiclive/transformers_cache
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="/admin/home-jordiclive/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/admin/home-jordiclive/Open-Assistant/model:$PYTHONPATH"

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES
hostfile="hostfile.txt"
rm -f $hostfile
for node in $HOSTNAMES; do
  echo $node slots=8 >> $hostfile
done


#export ADDR="$(hostname -f):29500"
#export MASTER_PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
cd /admin/home-jordiclive/Open-Assistant/model/model_training/
#srun deepspeed --hostfile $DEEPSPEED_HOSTFILE /admin/home-jordiclive/Open-Assistant/model/model_training/trainer_sft.py --configs defaults oasst_export_eu gpt-neox --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/output_dir --num_train_epochs 8 --residual_dropout 0.2 --deepspeed --num_train_epochs 12 --gradient_accumulation_steps 1 --use_flash_attention false --residual_dropout 0.0 --learning_rate 4e-6

deepspeed --master_addr $MASTER_ADDR --hostfile='/fsx/home-kaizhaol/long-context-transformers/hostfile.txt' --launcher OpenMPI /admin/home-jordiclive/Open-Assistant/model/model_training/trainer_sft.py --configs defaults oasst_export_eu llama-7b --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/output_dir --deepspeed --residual_dropout 0.0 --learning_rate 4e-6 --use_flash_attention False


#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g80"
#SBATCH --job-name=OA
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out

#source /admin/home-jordiclive/jordan/bin/activate open_assistant
#cd /admin/home-jordiclive/Open-Assistant
#export TRANSFORMERS_CACHE=/admin/home-jordiclive/transformers_cache
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export PYTHONPATH="/admin/home-jordiclive/Open-Assistant/model/model_training:$PYTHONPATH"
##export MASTER_PORT="12802"
##master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
##export MASTER_ADDR=$master_addr"i"
##echo "MASTER_ADDR="$MASTER_ADDR
##export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
#
#cd /admin/home-jordiclive/Open-Assistant/model/model_training/
#srun deepspeed /admin/home-jordiclive/Open-Assistant/model/model_training/trainer_sft.py --configs defaults oasst_export_eu gpt-neox --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/output_dir --num_train_epochs 8 --residual_dropout 0.2 --deepspeed --num_train_epochs 12 --gradient_accumulation_steps 1 --use_flash_attention false --residual_dropout 0.0 --learning_rate 4e-6

#def get_deepspeed_main_args(self):
#
#        args_list = list()
#
#        # get deepspeed runner args, and only pass them in to deepspeed launcher if they differ from defaults
#        for key, default_value in NeoXArgsDeepspeedRunner().defaults():
#            configured_value = getattr(self, key)
#            if configured_value != default_value:
#                args_list.extend(
#                    self.convert_key_value_to_command_line_arg(key, configured_value)
#                )
#        if 'DLTS_HOSTFILE' in os.environ:
#            args_list.extend(
#                self.convert_key_value_to_command_line_arg("hostfile", os.environ['DLTS_HOSTFILE'])
#            )