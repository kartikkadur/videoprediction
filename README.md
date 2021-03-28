# Video Prediction and Reconstruction
## A deep neural network model for video prediction derived from SDCNet for YoutubeVOS dataset.

Video Prediction is the task of estimating future frames given the current frame. The repository used SDCNet to train the video prediction model on
YoutubeVOS 2018 dataset. 

## Pretrained weights
The pretrained weights can be downloaded from the links below:
* Flownet2 Checkpoint : https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing
* Video Reconstruction model checkpoint for YoutubeVOS2018 dataset : coming soon

## Running Evaluation
To run the evaluation, do the following
* Download the pretrained weights for the YoutubeVOS dataset.
* Edit _eval.sh file with the pretrained weights paths for FLowNet2 and SDCNet.
* Run the script file once edited.
```
bash ./_eval.sh
```

## Video Reconstruction
The repository also supports Video Reconstruction for generating furtre frames for the Youtube VOS dataset.
To run the video reconstruction model, do the following
* Download the pretrained weights for the video reconstruction model and FLowNet2.
* Edit _aug.sh file with the pretrained weights paths for FLowNet2 and video reconstruction.
* Edit other paths to your preference.
* Run the _aug.sh file
```
bash ./_aug.sh
```

## Train the model
There is a shell script file for training the model (_train.sh). Edit the required paths in the script and run:
```
bash _train.sh
```

## References
The code in this repository is derived from https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet with custom additions for training on Youtube VOS dataset. Please refer that repository for training on other video datasets.

If you are interested in knowing more about the model please refer this paper : https://nv-adlr.github.io/publication/2018-Segmentation
