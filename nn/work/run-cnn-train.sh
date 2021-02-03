#!/bin/bash

# Configs with keywords to be sed 
classCount=##classes##
trainPoints=##train_points##
valPoints=##validation_points##
epochs=##epochs##
trainFile=##train_path##

# Path to CSVs
csvs="$HOME/cleanLearning/data/work"

ckt=${trainFile}

# Cleanup
rm -rf newPkl newNN
mkdir -p newPkl
mkdir -p newNN

cmd='wc -l <'

out="$(basename $ckt)"
# gets ckt name 
IFS='_' read -r -a array <<< "$out"
echo "Generating CNN and validation pkl for circuit ${array[0]}..."

# builds node embedding filename
embed=$csvs'/'${array[0]}'_node_embed.csv'
feat=$csvs'/'${array[0]}'_feat_sweep.csv'

pkl=${array[0]}'NodeCut'
data=${array[0]}'_data'
cnn=${array[0]}'_cnn'
infNc=${array[0]}'_infNc'
infData=${array[0]}'_infData'

dataPoints=$(eval "$cmd $ckt")
echo "There are $dataPoints data points"
echo "Training points $trainPoints; and validation points $valPoints"

# generates the nodeCut pkl 
python step0_genNodeCut.py $classCount newPkl/$pkl.pkl \
	$embed \
	$ckt
# generates pkl with splited training and validation data
python step1_genTrainValData.py $classCount newPkl/${pkl}.pkl $trainPoints $valPoints newPkl/${data}.pkl
# trains the CNN model
python step2_genCNN2.py newPkl/${data}.pkl $epochs newNN/${cnn}.ckpt 6

#python step3_genInferenceNodeCut.py $classCount newPkl/${infNc}.pkl \
#	$embed \
#	$feat

#python step4_genInferenceData.py $classCount newPkl/${infNc}.pkl newPkl/${infData}.pkl
