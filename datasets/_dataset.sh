#!/usr/bin/env bash

srun -K --ntasks=$1 --gpus-per-task=1 --cpus-per-gpu=6 -p V100-16GB --mem-per-gpu=48G\
  --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
  --container-image=/netscratch/enroot/dlcc_semantic-segmentation_20.06-py3.sqsh \
  --container-workdir=`pwd` \
  --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
  python frame_loader.py --sample 1