import sys
import os
import pickle
sys.path.append(os.path.abspath("../src"))
from NodeCut import NodeCut
################################################################################
## Configs
numClasses=int(sys.argv[1])
namePklFile=str(sys.argv[2])
print("################################################################################")
print("Starting Inference NodeCut generation with following variables:")
print("  numClasses  = %s" % str(numClasses))
print("  namePklFile = %s" % namePklFile)
################################################################################
## Read CSV files
nc=NodeCut(numClasses=numClasses,train=False)
print("  Reading CSV files")
for argIdx in range(3,len(sys.argv)):
	cktId = int((argIdx - 3) / 2)
	print("    Reading %s of cktId %d" % (sys.argv[argIdx], cktId))
	if argIdx % 2 != 0:
		nc.readEmbed(sys.argv[argIdx], cktId)
	else:
		nc.readCSV(sys.argv[argIdx], cktId)
################################################################################
# # Plot data correlation
# nc.plotCorr()
################################################################################
# Save pkl file
print("  Saving NodeCut to %s" % namePklFile)
with open(namePklFile, 'wb') as f:
	pickle.dump(nc, f, pickle.HIGHEST_PROTOCOL)
	f.close()
