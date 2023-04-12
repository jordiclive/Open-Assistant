#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g40"
#SBATCH --job-name=OA
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out
#SBATCH --open-mode=append
#SBATCH --exclusive

module load openmpi
#module load cuda/11.7
module purge
module pdsh



export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export OMPI_MCA_mtl_base_verbose=1
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

export NCCL_DEBUG=WARN
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without these set; See https://github.com/NVIDIA/nccl/issues/676
# export NCCL_P2P_DISABLE=1
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME="eth0"


export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export OMPI_MCA_btl="^openib"

echo go $COUNT_NODE
echo $HOSTNAMES
hostfile="/admin/home-jordiclive/Open-Assistant/hostfile.txt"
rm -f $hostfile
for node in $HOSTNAMES; do
  echo $node slots=8 >> $hostfile
done


cd /admin/home-jordiclive/Open-Assistant
export TRANSFORMERS_CACHE=/fsx/home-jordiclive/transformers_cache
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="/admin/home-jordiclive/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/admin/home-jordiclive/Open-Assistant/model:$PYTHONPATH"
#export ADDR="$(hostname -f):29500"
#export MASTER_PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
cd /admin/home-jordiclive/Open-Assistant/model/model_training/
#srun deepspeed --hostfile $DEEPSPEED_HOSTFILE /admin/home-jordiclive/Open-Assistant/model/model_training/trainer_sft.py --configs defaults oasst_export_eu gpt-neox --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/output_dir --num_train_epochs 8 --residual_dropout 0.2 --deepspeed --num_train_epochs 12 --gradient_accumulation_steps 1 --use_flash_attention false --residual_dropout 0.0 --learning_rate 4e-6
pip install mpi4py
source /fsx/home-jordiclive/miniconda3/bin/activate open

deepspeed --num_nodes -1 --launcher openmpi --master_addr $MASTER_ADDR --hostfile=$hostfile /admin/home-jordiclive/Open-Assistant/model/model_training/trainer_sft.py --configs defaults oasst_export_eu llama-7b --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/output_dir --deepspeed --residual_dropout 0.0 --learning_rate 4e-6 --use_flash_attention False

