#!/bin/bash
# Configs
classCount=10
trainPoints=50000
valPoints=10000
epochs=50
python step3_genInferenceNodeCut.py $classCount pkl/arbiter_infNodeCut.pkl \
			../../data/work/arbiter_embed.csv \
    	../../data/work/arbiter_features.csv

python step4_genInferenceData.py $classCount pkl/arbiter_infNodeCut.pkl pkl/arbiter_infData.pkl

python step5_inference.py pkl/arbiter_infData.pkl cpCNN/newRcCla.ckpt
