#!/bin/bash
# Configs
classCount=10
trainPoints=50000
valPoints=10000
epochs=50

# Path to CSVs
csvs="$HOME/cleanLearning/data/work"

#rm -f test_classification.txt
cmd='wc -l <'
for ckt in ${csvs}/new_nn*_hashed.csv;
do
	# gets ckt name 
	IFS='_' read -r -a array <<< "$ckt"
	echo "Generating validation pkl for circuit ${array[2]}..."

	# builds node embedding filename
	embed=$csvs'/'${array[2]}'_node_embed.csv'
	feat=$csvs'/'${array[2]}'_feat_sweep.csv'

	pkl=${array[2]}'NodeCut'
	data=${array[2]}'_validation_data'
	#cnn=${array[2]}'_cnn'
	#infNc=${array[2]}'_infNc'
	#infData=${array[2]}'_infData'

	trainPoints=$(eval "$cmd $ckt")
	# gets out the csv header file
	trainPoints=$((trainPoints-51))
	
	# python step0_genNodeCut.py $classCount pkl/$pkl.pkl \
	# 				$embed \
	# 				$ckt

	python step1_genTrainValData.py $classCount pkl/${pkl}.pkl 50 $trainPoints pkl/${data}.pkl
	#python step2_genCNN.py pkl/${data}.pkl $epochs cpCNN/${cnn}.ckpt

	# python step3_genInferenceNodeCut.py $classCount pkl/${infNc}.pkl \
	# 			$embed \
	# 			$feat

#	python step4_genInferenceData.py $classCount pkl/${infNc}.pkl pkl/${infData}.pkl

done

# Generates NodeCut with $classCount classes and saves to pkl/NodeCut.pkl using CSV files
# python step0_genNodeCut.py $classCount pkl/NodeCut.pkl \
# 		../../data/work/rc16b_node_embed.csv \
# 		../../data/work/new_nn_rc16b_hashed.csv \
# 		../../data/work/mul4b_node_embed.csv \
#	 	../../data/work/new_nn_mul4b_hashed.csv 
# 	../../data/work/bfly_node_embed.csv \
# 	../../data/work/new_nn_bfly_comb_hashed.csv \
# 	../../data/work/c1908_node_embed.csv \
# 	../../data/work/new_nn_c1908_comb_hashed.csv \
# 	../../data/work/c880_node_embed.csv \
# 	../../data/work/new_nn_c880_comb_hashed.csv \
#
# 	../../data/work/des_node_embed.csv \
# 	../../data/work/new_nn_des_comb_hashed.csv \


	# ../../data/work/new_nn_ode_comb_hashed.csv \
	# ../../data/work/ode_node_embed.csv

# Generates Training and Validation data using NodeCut with $classCount classes, $trainPoints training points and $valPoints validation points, saving to pkl/data.pkl
#python step1_genTrainValData.py $classCount pkl/NodeCut.pkl $trainPoints $valPoints pkl/data.pkl
# Creates Neural Network, trains using pkl/data.pkl for $epochs epochs and stores checkpoint to cpCNN/cnn4.ckpt
#python step2_genCNN.py pkl/data.pkl $epochs cpCNN/cnn4.ckpt $optThreshold
# Generates Inference NodeCut with $classCount classes and saves to pkl/infNodeCut.pkl using CSV files
# python step3_genInferenceNodeCut.py $classCount pkl/infNodeCut.pkl \
# 			../../data/work/sqrt_embed.csv \
#     	../../data/work/sqrt_feat.csv
# # Generates Inference data using infNodeCut with $classCount classes,saving to pkl/infData.pkl
#python step4_genInferenceData.py $classCount pkl/infNodeCut.pkl pkl/infData.pkl
# # Makes inferences using model stored at cpCNN/cnn4.ckpt using data from pkl/infData.pkl
#python step5_inference.py pkl/infData.pkl cpCNN/cnn4.ckpt
