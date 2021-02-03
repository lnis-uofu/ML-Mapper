import sys
import os
import numpy as np
import pickle
import tensorflow as tf
import plotly.graph_objects as go
sys.path.append(os.path.abspath("../src"))
from CNN import CNN
################################################################################
## Configs
dataPklFile=str(sys.argv[1])
epochs=int(sys.argv[2])
checkPointPath=str(sys.argv[3])
#optThreshold=int(sys.argv[4])
print("################################################################################")
print("Starting CNN generation with following variables:")
print("  dataPklFile    = %s" % dataPklFile)
print("  epochs         = %s" % str(epochs))
print("  checkPointPath = %s" % checkPointPath)
#print("  optThreshold   = %s" % str(optThreshold))
################################################################################
# Loads data
print("  Loading data from %s " % dataPklFile)
if os.path.exists(dataPklFile):
	with open(dataPklFile, 'rb') as f:
		dataDict = pickle.load(f)
	f.close()
trainFeatureNPArray = dataDict["train"]["features"]
trainLabelNPArray = dataDict["train"]["labels"]
valFeatureNPArray = dataDict["val"]["features"]
valLabelNPArray = dataDict["val"]["labels"]
featureShape = dataDict["config"]["featureShape"]
numClasses = dataDict["config"]["numClasses"]
occurList = []
for idx in range(10):
    occur = list(trainLabelNPArray).count(idx)
    occurList.append(occur)
print("    Training classes are (%d): %s" % (len(list(trainLabelNPArray)), str(occurList)))
occurList = []
for idx in range(10):
    occur = list(valLabelNPArray).count(idx)
    occurList.append(occur)
print("    Validation classes are (%d): %s" % (len(list(valLabelNPArray)), str(occurList)))
################################################################################
# Create neural network
print("  Creating Neural Network")
# Create CNN
cnn = CNN(featureShape=featureShape, numClasses=numClasses)
# Print summary
cnn.model.summary()
################################################################################
# Train neural network
print("  Training Neural Network")
for trainFeatureNP in trainFeatureNPArray:
    if np.any(np.isnan(trainFeatureNP)):
        print(trainFeatureNP)
# print(len(trainFeatureNPArray))
# print(trainFeatureNPArray[0])
# print(trainLabelNPArray[0])
# Checkpoint callback
checkPointCallBack = tf.keras.callbacks.ModelCheckpoint(filepath=checkPointPath,
                                                        save_weights_only=True,
                                                        verbose=1)
history = cnn.model.fit(trainFeatureNPArray,trainLabelNPArray,
                        epochs=epochs, verbose=2,
                        validation_data=(valFeatureNPArray,valLabelNPArray),
                        callbacks=[checkPointCallBack])
# Plot training info
lossListY = history.history["loss"]
lossListX = list(range(len(lossListY)))
valLossListY = history.history["val_loss"]
valLossListX = list(range(len(valLossListY)))
# fig0 = go.Figure(data=[go.Scatter(x=lossListX,y=lossListY,name="Training Loss"),
#                        go.Scatter(x=valLossListX,y=valLossListY,name="Validation Loss")])
# fig0.show()

################################################################################
# Checks training
#print("  Making inferences")
#inferences = cnn.model.predict(valFeatureNPArray)
#inferences = tf.argmax(inferences, 1).numpy()

#rightInf = 0
#totalInf = 0
#for inf, label in zip(inferences, valLabelNPArray):
#    totalInf += 1
#    if ((inf <= optThreshold) and (label <= optThreshold)) or ((inf > optThreshold) and (label > optThreshold)):
#        rightInf += 1
#print("    Performed %d inferences with accuracy of %f" % (totalInf, rightInf/totalInf))
