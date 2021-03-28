#!/usr/bin/env bash
TRAIN_FILE=/ds/videos/YoutubeVOS2018/train_all_frames
VAL_FILE=/ds/videos/YoutubeVOS2018/valid_all_frames
SDC2DREC_CHECKPOINT=/netscratch/kadur/lable_propagation/semantic-segmentation-sdcnet/sdcnet/EXP1_SeqLen5/_ckpt_epoch_270_iter_0013499.pth
FLOWNET2_CHECKPOINT=../pretrained_models/FlowNet2_checkpoint.pth.tar
SEQLENGTH=5
SAVENAME=EXP1_SeqLen5
SAVEDIR=./
EPOCHS=400
DATASET=FrameLoader
MODEL=SDCNet2DRecon
MODELSAVE=10
BATCHSIZE=224
LEARNINGRATE=1e-5
WEIGHT_DECAY=0
STRIDE=64
VAL_FREQ=25
PRINTFREQ=100
START_EPOCH=270
SAMPLE=1
CROPSIZE=256
SAMPLE_RATE=5
LRSCHEDULER=MultiStepLR
 
srun -K --ntasks=$1 --gpus-per-task=1 --cpus-per-gpu=6 -p V100-32GB --mem-per-gpu=48G\
  --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
  --container-image=/netscratch/enroot/dlcc_semantic-segmentation_20.06-py3.sqsh \
  --container-workdir=`pwd` \
  --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
  python main.py \
  --sequence_length ${SEQLENGTH} \
  --save ${SAVEDIR} \
  --name ${SAVENAME} \
  --epochs ${EPOCHS} \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --train_file ${TRAIN_FILE} \
  --val_file ${VAL_FILE} \
  --save_freq ${MODELSAVE} \
  --flownet2_checkpoint ${FLOWNET2_CHECKPOINT} \
  --batch_size ${BATCHSIZE} \
  --optimizer Adam \
  --lr ${LEARNINGRATE} \
  --stride ${STRIDE} \
  --print_freq ${PRINTFREQ} \
  --start_epoch ${START_EPOCH} \
  --lr_scheduler ${LRSCHEDULER} \
  --val_batch_size ${BATCHSIZE} \
  --val_freq ${VAL_FREQ} \
  --sample ${SAMPLE} \
  --wd ${WEIGHT_DECAY} \
  --sample_rate ${SAMPLE_RATE} \
  --resume ${SDC2DREC_CHECKPOINT} \
  
