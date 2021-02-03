import sys
import os
import pickle
import tensorflow as tf
sys.path.append(os.path.abspath("../src"))
from CNN import CNN

tf.enable_eager_execution()
################################################################################
## Configs
dataPklFile=str(sys.argv[1])
checkPointPath=str(sys.argv[2])
print("################################################################################")
print("Starting CNN generation with following variables:")
print("  dataPklFile    = %s" % dataPklFile)
print("  checkPointPath = %s" % checkPointPath)
################################################################################
# Loads data

pklFile = os.path.basename(dataPklFile)
cktName = pklFile.split("_")
infFile = cktName[0] + "_inf.txt"

print("  Loading data from %s " % dataPklFile)
if os.path.exists(dataPklFile):
	with open(dataPklFile, 'rb') as f:
		dataDict = pickle.load(f)
	f.close()
infFeatureNPArray = dataDict["data"]["features"]
infIdList = dataDict["data"]["nodeId"]
infCutIdList = dataDict["data"]["cutIds"]
featureShape = dataDict["config"]["featureShape"]
numClasses = dataDict["config"]["numClasses"]

################################################################################
# Reloads neural network
print("  Reloading Neural Network")
# Create CNN
cnn = CNN(featureShape=featureShape, numClasses=numClasses)
# Print summary
cnn.model.summary()
# Loads the weights
cnn.model.load_weights(checkPointPath)
################################################################################
# Makes inferences
print("  Making inferences")
inferences = cnn.model.predict(infFeatureNPArray)
inferenceList = list(tf.argmax(inferences, 1).numpy())

# print("Feature array")
# print(infFeatureNPArray.tolist())
# print("Label array")
# print(inferenceList)
# print("Id and cut list")
# print(infIdList)
# print(infCutIdList)

print("File name: " + infFile)
f = open(infFile, "w")
nodeDict = {}
used = 0
notUsed = 0
for infId, infCutId, infClass in zip(infIdList, infCutIdList, inferenceList):
	if infClass <= 3:
		used = used + 1
		if infId in nodeDict:
			nodeDict[infId].append(infCutId)
		else:
			nodeDict[infId] = [infCutId]
	#notUsed = notUsed + 1
	elif infId not in nodeDict and infClass <= 6:
		used = used + 1
		if infId in nodeDict:
		 	nodeDict[infId].append(infCutId)
		else:
			nodeDict[infId] = [infCutId]
			continue
	
	elif infId in nodeDict and len(nodeDict[infId]) == 1 and infClass <= 6:
		used = used + 1
		nodeDict[infId].append(infCutId)
	notUsed = notUsed + 1 
print("Used " + str(used) + " cuts; total " + str(notUsed) + " cuts")
for key in nodeDict:
	nodeId = str(key)
	for element in nodeDict[key]:
		nodeId += ", " + str(element)
	nodeId += "\n"
	f.write("%s" % nodeId)
f.close()
#optThreshold = 6
#rightInf = 0
#totalInf = 0
#for inf, label in zip(inferenceList, valLabelNPArray):
#	totalInf += 1
#	if ((inf <= optThreshold) and (label <= optThreshold)) or ((inf > optThreshold) and (label > optThreshold)):
#		rightInf += 1
#print("    Performed %d inferences with accuracy of %f" % (totalInf, rightInf/totalInf))
