#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g40"
#SBATCH --job-name=OA
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --output=%x_%j.out
#SBATCH --exclusive


module purge
module load openmpi
module load cuda/11.7


#mkdir -p /mnt/nvme/home/$(whoami)/hostfiles
#hostfile=/mnt/nvme/home/$(whoami)/hostfiles/hosts_$SLURM_JOBID
#rm $hostfile &> /dev/null # for consecutive calls to this script in interactive jobs
#
#
#
#hostfile = "/fsx/home-jordiclive/hostfile.txt"
#for i in `scontrol show hostnames $SLURM_NODELIST`
#do
#    echo $i slots=8 >>$hostfile
#done



export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

hostfile="/admin/home-jordiclive/Open-Assistant/hostfile.txt"
rm -f $hostfile
for node in $HOSTNAMES; do
  echo $node slots=8 >> $hostfile
done

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib
#export NCCL_PROTO=simple
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/aws-ofi-nccl/lib
#export PATH=$PATH:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin
#export FI_EFA_FORK_SAFE=1
#export FI_LOG_LEVEL=1
#export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
#export NCCL_DEBUG=info
#export OMPI_MCA_mtl_base_verbose=1
#export FI_EFA_ENABLE_SHM_TRANSFER=0
#export FI_PROVIDER=efa
#export FI_EFA_TX_MIN_CREDITS=64
#export NCCL_TREE_THRESHOLD=0
#export OMPI_MCA_pml="^cm"
#export OMPI_MCA_btl="tcp,self"
#export OMPI_MCA_btl_tcp_if_exclude="lo,docker1"
#export OMPI_MCA_plm_rsh_no_tree_spawn=1
#export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
#export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
#export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"
#
#export NCCL_DEBUG=WARN
#export NCCL_TREE_THRESHOLD=0
#export NCCL_PROTO=simple
## Network issues without these set; See https://github.com/NVIDIA/nccl/issues/676
## export NCCL_P2P_DISABLE=1
#export NCCL_IBEXT_DISABLE=1
#export NCCL_SOCKET_IFNAME="eth0"
#
#
#export FI_EFA_FORK_SAFE=1
#export FI_LOG_LEVEL=1
#export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
#export FI_EFA_ENABLE_SHM_TRANSFER=0
#export FI_PROVIDER=efa
#export FI_EFA_TX_MIN_CREDITS=64
#
#export PYTHONFAULTHANDLER=1

# export CUDA_LAUNCH_BLOCKING=1

export OMPI_MCA_mtl_base_verbose=1



export OMPI_MCA_mtl_base_verbose=1
#export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH:/usr/lib64/compat-openmpi16/lib
#export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
#
#export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without these set; See https://github.com/NVIDIA/nccl/issues/676
# export NCCL_P2P_DISABLE=1
export NCCL_IBEXT_DISABLE=1
#export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_HCA=ibp
export NCCL_SOCKET_IFNAME="^lo,docker0"
export NCCL_IB_HCA=ibp
export NCCL_COLLNET_ENABLE=1
#export TORCH_EXTENSIONS_DIR=extensions

export OMPI_MCA_btl="^openib"
source /fsx/home-jordiclive/miniconda3/bin/activate open

cd /admin/home-jordiclive/Open-Assistant/model/model_training/
export TRANSFORMERS_CACHE=/fsx/home-jordiclive/transformers_cache
export PYTHONPATH="/admin/home-jordiclive/Open-Assistant/model/model_training:$PYTHONPATH"
export PYTHONPATH="/admin/home-jordiclive/Open-Assistant/model:$PYTHONPATH"
export DLTS_HOSTFILE=$hostfile


deepspeed --num_nodes -1 --launcher openmpi --hostfile=$hostfile --master_addr $MASTER_ADDR  /admin/home-jordiclive/Open-Assistant/model/model_training/trainer_sft.py --configs defaults oasst_export_eu llama-7b --cache_dir /fsx/home-jordiclive/data_cache --output_dir /fsx/home-jordiclive/output_dir --deepspeed --residual_dropout 0.0 --learning_rate 4e-6 --use_flash_attention False


