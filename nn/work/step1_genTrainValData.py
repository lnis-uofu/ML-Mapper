import sys
import os
import pickle
import numpy as np
sys.path.append(os.path.abspath("../src"))
from NodeCut import NodeCut
################################################################################
## Configs
numClasses=str(sys.argv[1])
nodeCutPklFile=str(sys.argv[2])
trainingPoints=int(sys.argv[3])
validationPoints=int(sys.argv[4])
dataPklFile=str(sys.argv[5])
print("################################################################################")
print("Starting data generation with following variables:")
print("  numClasses       = %s" % numClasses)
print("  nodeCutPklFile   = %s" % nodeCutPklFile)
print("  trainingPoints   = %s" % str(trainingPoints))
print("  validationPoints = %s" % str(validationPoints))
print("  dataPklFile      = %s" % dataPklFile)
################################################################################
# Loads NodeCut
print("  Loading NodeCut from %s " % nodeCutPklFile)
if os.path.exists(nodeCutPklFile):
	with open(nodeCutPklFile, 'rb') as f:
		nc = pickle.load(f)
	f.close()
################################################################################
# Prepare data
print("  Preparing data")
nc.prepare(numTrainPoints=trainingPoints, numValPoints=validationPoints, balanced=False)
# from tabulate import tabulate
# print(tabulate(nc.nodeEmbedDf, headers='keys', tablefmt='psql'))

################################################################################
# Plot data correlation
# print("  Plotting training data classes correlation")
# nc.plotTrainCorr()

################################################################################
# Prepare training data
print("  Collecting training features and labels")
featureList, labelList, idList, cutIdList = nc.getTrainFeatureLabelTuple()
# print(featureList[0])
# os.sys(exit)

trainFeatureList = featureList

trainFeatureNPArray = nc.reshapeFeature(np.array(trainFeatureList))
trainLabelNPArray = np.array(labelList)
trainIdListNPArray = np.array(idList)
trainCutIdListNPArray = np.array(cutIdList)

# print("Feature array")
# print(trainFeatureNPArray.tolist())
# print("Label array")
# print(trainLabelNPArray.tolist())
# print("Id and cut list")
# print(trainIdListNPArray.tolist())
# print(trainCutIdListNPArray.tolist())

print("  Read %d training data points" % len(featureList))

occurList = []
for idx in range(10):
	occur = labelList.count(idx)
	occurList.append(occur)
print("    Classes are (%d): %s" % (len(labelList), str(occurList)))

print("  Collecting validation features and labels")
featureList, labelList, idList, cutIdList = nc.getValFeatureLabelTuple()
valFeatureList = featureList
valFeatureNPArray = nc.reshapeFeature(np.array(valFeatureList))
valLabelNPArray = np.array(labelList)
valIdListNPArray = np.array(idList)
valCutIdListNPArray = np.array(cutIdList)

print("  Read %d validation data points" % len(featureList))

occurList = []
for idx in range(10):
	occur = labelList.count(idx)
	occurList.append(occur)
print("    Classes are (%d): %s" % (len(labelList), str(occurList)))
################################################################################
# Save data pkl file
print("  Saving data as dictionary to %s" % dataPklFile)
dataDict = {
	"config" : {
		"featureShape" : nc.getFeatureShape(),
		"numClasses" : numClasses
	},
	"train" : {
		"features" : trainFeatureNPArray,
		"labels" : trainLabelNPArray,
		"nodeId" : trainIdListNPArray,
		"cutIds" : trainCutIdListNPArray
	},
	"val" : {
		"features" : valFeatureNPArray,
		"labels" : valLabelNPArray,
		"nodeId" : valIdListNPArray,
		"cutIds" : valCutIdListNPArray
	}
}
with open(dataPklFile, 'wb') as f:
	pickle.dump(dataDict, f, pickle.HIGHEST_PROTOCOL)
	f.close()
