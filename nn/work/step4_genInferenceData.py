import sys
import os
import pickle
import numpy as np
sys.path.append(os.path.abspath("../src"))
from NodeCut import NodeCut
################################################################################
## Configs
numClasses=str(sys.argv[1])
infNodeCutPklFile=str(sys.argv[2])
infDataPklFile=str(sys.argv[3])
print("################################################################################")
print("Starting inference data generation with following variables:")
print("  numClasses        = %s" % numClasses)
print("  infNodeCutPklFile = %s" % infNodeCutPklFile)
print("  infDataPklFile    = %s" % infDataPklFile)
################################################################################
# Loads NodeCut
print("  Loading Inference NodeCut from %s " % infNodeCutPklFile)
if os.path.exists(infNodeCutPklFile):
	with open(infNodeCutPklFile, 'rb') as f:
		nc = pickle.load(f)
	f.close()
################################################################################
# Prepare data
print("  Preparing data")
nc.prepare(numTrainPoints=0, balanced=False)
print("  Collecting features and labels")
featureList, labelList, idList, cutIdList =nc.getValFeatureLabelTuple()
infFeatureList = featureList
infFeatureNPArray = nc.reshapeFeature(np.array(infFeatureList))
print("  Read %d data points" % len(featureList))
################################################################################
# Save data pkl file
print("  Saving data as dictionary to %s" % infDataPklFile)
dataDict = {
	"config" : {
		"featureShape" : nc.getFeatureShape(),
		"numClasses" : numClasses
	},
	"data" : {
		"features" : infFeatureNPArray,
		"nodeId" : idList,
		"cutIds" : cutIdList
	}
}
with open(infDataPklFile, 'wb') as f:
	pickle.dump(dataDict, f, pickle.HIGHEST_PROTOCOL)
	f.close()
