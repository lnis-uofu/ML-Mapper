#!/bin/bash
# Configs
classCount=10
trainPoints=50000
valPoints=10000
epochs=50
# Generates NodeCut with $classCount classes and saves to pkl/NodeCut.pkl using CSV files
#python step0_genNodeCut.py $classCount pkl/MulRcNodeCut.pkl \
# 		../../data/work/rc16b_node_embed.csv \
# 		../../data/work/new_nn_rc16b_hashed.csv \
# 		../../data/work/mul4b_node_embed.csv \
# 		../../data/work/new_nn_mul4b_hashed.csv 



# Generates Training and Validation data using NodeCut with $classCount classes, $trainPoints training points and $valPoints validation points, saving to pkl/data.pkl
#python step1_genTrainValData.py $classCount pkl/MulRcNodeCut.pkl 100000 30000 pkl/MulRcdata.pkl
# Creates Neural Network, trains using pkl/data.pkl for $epochs epochs and stores checkpoint to cpCNN/cnn4.ckpt
python step2_genCNN2.py pkl/MulRcdata.pkl $epochs cpCNN/MulRcCnn.ckpt 6
# Generates Inference NodeCut with $classCount classes and saves to pkl/infNodeCut.pkl using CSV files
#python step3_genInferenceNodeCut.py $classCount pkl/booth64_infNodeCut.pkl \
#			../../data/work/booth64_embed.csv \
#    	../../data/work/booth64_feat.csv
# # Generates Inference data using infNodeCut with $classCount classes,saving to pkl/infData.pkl
#python step4_genInferenceData.py $classCount pkl/booth64_infNodeCut.pkl pkl/booth64_infData.pkl
# # Makes inferences using model stored at cpCNN/cnn4.ckpt using data from pkl/infData.pkl
#python step5_inference.py pkl/booth64_infData.pkl cpCNN/rc16b_cnn.ckpt
