srun -K --ntasks=$1 --gpus-per-task=1 --cpus-per-gpu=6 --nodes=1 -p V100-32GB --mem-per-gpu=48G\
  --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
  --container-image=/netscratch/enroot/dlcc_semantic-segmentation_20.06-py3.sqsh \
  --container-workdir=`pwd` \
  --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
	python sdc_aug.py --propagate 5 --vis \
		--flownet2_checkpoint ../pretrained_models/FlowNet2_checkpoint.pth.tar \
		--pretrained /netscratch/kadur/lable_propagation/semantic-segmentation-sdcnet/sdcnet/EXP6_SeqLen2_BGR/_ckpt_epoch_390_iter_0176279.pth \
		--source_dir /ds/videos/YoutubeVOS2018 \
		--target_dir /netscratch/kadur/lable_propagation/augmentation/YoutubeVOS2018_PL4 \
		--sequence_length 2 \
		--propagate 4 \
		#--scene ffc43fc345 \
		


		
