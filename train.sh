#!/bin/bash
python train.py --lr 0.001 --batch-size 80 --arch resnet18 \
 	--data-name hmdb51 --representation residual \
 	--data-root data/hmdb51/mpeg4_videos \
 	--train-list data/datalists/hmdb51_split1_train.txt \
 	--test-list data/datalists/hmdb51_split1_test.txt \
 	--model-prefix hmdb51_residual_model \
 	--lr-steps 120 180 240  --epochs 300 \
 	--gpus 0 \
    --weights hmdb51_residual_model_residual_checkpoint.pth.tar > train.log

cp hmdb* ../residual/21-0801/
mv train.log ../residual/21-0801/

python train.py --lr 0.001 --batch-size 80 --arch resnet18 \
 	--data-name hmdb51 --representation residual \
 	--data-root data/hmdb51/mpeg4_videos \
 	--train-list data/datalists/hmdb51_split2_train.txt \
 	--test-list data/datalists/hmdb51_split2_test.txt \
 	--model-prefix hmdb51_residual_model \
 	--lr-steps 120 180 240  --epochs 300 \
 	--gpus 0 \
    --weights hmdb51_residual_model_residual_checkpoint.pth.tar > train.log

cp hmdb* ../residual/22-0801/
mv train.log ../residual/22-0801/

python train.py --lr 0.001 --batch-size 80 --arch resnet18 \
 	--data-name hmdb51 --representation residual \
 	--data-root data/hmdb51/mpeg4_videos \
 	--train-list data/datalists/hmdb51_split3_train.txt \
 	--test-list data/datalists/hmdb51_split3_test.txt \
 	--model-prefix hmdb51_residual_model \
 	--lr-steps 120 180 240  --epochs 300 \
 	--gpus 0 \
    --weights hmdb51_residual_model_residual_checkpoint.pth.tar > train.log

cp hmdb* ../residual/23-0801/
mv train.log ../residual/23-0801/