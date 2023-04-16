#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g80"
#SBATCH --job-name=OA
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out
#SBATCH --exclusive


module purge
module load openmpi
module load cuda/11.7


#mkdir -p /mnt/nvme/home/$(whoami)/hostfiles#
#for i in `scontrol show hostnames $SLURM_NODELIST`
#do
#    echo $i:8 >>$machinefile
#done


hostfile='/admin/home-jordiclive/Open-Assistant/hostfile.txt'
#machinefile='/admin/home-jordiclive/Open-Assistant/machinefile.txt'
rm $hostfile  for consecutive calls to this script in interactive jobs
#

for i in `scontrol show hostnames $SLURM_NODELIST`
do
    echo $i slots=8 >>$hostfile
done

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`


export OMPI_MCA_mtl_base_verbose=1


export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without these set; See https://github.com/NVIDIA/nccl/issues/676
# export NCCL_P2P_DISABLE=1
export NCCL_IBEXT_DISABLE=1
#export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_HCA=ibp
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export NCCL_COLLNET_ENABLE=1
export NCCL_SOCKET_IFNAME="^lo,docker0"
export OMPI_MCA_btl="^openib"
#export TORCH_EXTENSIONS_DIR=extensions


source /admin/home-jordiclive/jordan_no/bin/activate

cd /admin/home-jordiclive/Open-Assistant/model/model_training/
export TRANSFORMERS_CACHE=/fsx/home-jordiclive/transformers_cache
export PYTHONPATH="/admin/home-jordiclive/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/admin/home-jordiclive/Open-Assistant/model:$PYTHONPATH"
export DLTS_HOSTFILE=$hostfile
export WANDB_API_KEY= 'd8216641d549f9bb3d0c5074baa39e15dfd55030'


deepspeed --launcher openmpi --hostfile '/admin/home-jordiclive/Open-Assistant/hostfile.txt' --master_addr $MASTER_ADDR  /admin/home-jordiclive/Open-Assistant/model/model_training/trainer_sft.py --configs llama-66b --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/66B_checkpoints/output_dir --deepspeed
