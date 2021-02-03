## \file NodeCut.py
#  \brief Data set of nodes and their possible cuts.
#
# Data set is modeled as a Pandas Dataframe. Data is added to data set in the
# form of CSV files. Each CSV file must contain a set of different
# implementations of a given circuit. As data is read to data set, labels
# are normalized to maximum label value in CSV. This way, the number of
# different implementations of a given circuit available in the CSV impacts how
# representative the data set is. CSV needs to contain enough different
# implementations with different QoR values. After reading CSV data, prepare the
# data set using the prepare method. After data, training and validation data
# can be collected using the getTrainFeatureLabelTuple and
# getValFeatureLabelTuple methods.
#
# Object constructor can be empty or receive the number of classes to use. To
# create default:
# * nc=NodeCut()
# To create data set to classify among 5 classes:
# * nc=NodeCut(numClasses=5)
#
# To add data, use readCSV method, providing a CSV file name:
# * nc.readCSV("myPath/data.csv")
#
# To prepare, use prepare method, providing number of training and validation
# points:
# * nc.prepare(numTrainPoints=300, numValPoints=700)
#
# To collect training data, use the getTrainFeatureLabelTuple method:
# * featureList, labelList=nc.getTrainFeatureLabelTuple()
#
# To collect validation data, use the getValFeatureLabelTuple method:
# * featureList, labelList=nc.getValFeatureLabelTuple()
#

import pandas as pd
import numpy as np
import plotly.express as px

class NodeCut():

	# Data set column name labels
	lCktId="cktid"
	lNodeId="nodeid"
	lNumFanout="fon"
	lNodeLevel="lvln"
	lNodeHasInversion="invn"
	lNodeRelativeLvl="relativelvl"
	lChild1HasInversion="invp1"
	lChild1Level="lvlp1"
	lChild1Fo="fop1"
	lChild2HasInversion="invp2"
	lChild2Level="lvlp2"
	lChild2Fo="fop2"
	lCutTruthTable="tt"
	lCutIsInverted="invc"
	lCutNumLeaves="leavesc"
	lCutVolume="volumec"
	lCutMinLvl="mincutlvl"
	lCutMaxLvl="maxcutlvl"
	lCutLvl="cutlvl"
	lCutMinFo="cutminfo"
	lCutMaxFo="cutmaxfo"
	lCutFo="cutfo"
	lCutIdx="cutidx"
	lCutImpl="gate"
	l1id="l1id"
	l2id="l2id"
	l3id="l3id"
	l4id="l4id"
	l5id="l5id"
	lCutDelay="delay"
	lDataType="dataType"
	# Data type labels
	lDataTypeTrain="train"
	lDataTypeVal="val"
	lDataTypeNone="none"
	lEmbedId="eid"
	lEmbedFo="efo"
	lEmbedLvl="elvl"
	lEmbedInv="einv"
	lEmbedC1Inv="ec1inv"
	lEmbedC1Lvl="ec1lvl"
	lEmbedC1Fo="ec1fo"
	lEmbedC2Inv="ec2inv"
	lEmbedC2Lvl="ec2lvl"
	lEmbedC2Fo="ec2fo"
	lEmbedRLvl="erelvl"

	## Constructor
	#
	# Creates object.
	#
	# \param self
	# \param train is int defining if object is used for training or not
	# \param numClasses is bool defining number of classes to use. Default is 4
	def __init__(self, numClasses=4, train=False):
		# Pointer to Pandas Dataframe
		self.__df = None
		# Controls if data set can still accept new CSV files
		self.__lock = False
		# Dictionary with node embedding
		self.__nodeEmbedDf = None
		# Stores max values used to normalize features
		self.__maxNumFanout=0
		self.__maxNodeLevel=0
		self.__maxNodeHasInversion=0
		self.__maxNodeRelativeLevel=0
		self.__maxChild1HasInversion=0
		self.__maxChild1Level=0
		self.__maxChild1Fo=0
		self.__maxChild2HasInversion=0
		self.__maxChild2Level=0
		self.__maxChild2Fo=0
		self.__maxCutTruthTable=0
		self.__maxCutIsInverted=0
		self.__maxCutNumLeaves=0
		self.__maxCutVolume=0
		self.__maxMinCutLvl=0
		self.__maxMaxCutLvl=0
		self.__maxCutLvl=0
		# Defines number of classes:
		self.__numClasses=numClasses
		# Stores object type
		self.__train=train

	## Reads a CSV file
	#
	# CSV is expected to have columns defined by data set column name labels.
	# Multiple CSV files can be read for single object. After reading in CSV
	# data, labels are normalized.
	#
	# \param self
	# \param fileName is a string defining the name of CSV file to read
	# \param cktId defines the id of the circuit being read
	def readCSV(self, fileName, cktId):
		if self.__lock:
			raise RuntimeError("Can't read new CSV file to NodeCut that was already locked by the \"prepare\" method")
		# Read the CSV
		df=pd.read_csv(fileName)
		# Remove the gate column
		# df=df.drop(['gate'], axis=1)
		df[NodeCut.lCktId] = cktId
		# Normalize labels
		if (self.train):
			df[[NodeCut.lCutDelay]]=(df[[NodeCut.lCutDelay]]-df[[NodeCut.lCutDelay]].min())/(df[[NodeCut.lCutDelay]].max()-df[[NodeCut.lCutDelay]].min())
		if self.__df is None:
			self.__df = df
		else:
			self.__df = self.__df.append(df, ignore_index=True)

	## Reads a CSV file with Node Embedding
	#
	# CSV is expected to have columns defined by data set column name labels.
	# Multiple CSV files can be read for single object. After reading in CSV
	# data, labels are normalized.
	#
	# \param self
	# \param fileName is a string defining the name of CSV file to read
	# \param cktId defines the id of the circuit being read
	def readEmbed(self, fileName, cktId):
		# if self.__lock:
		# 	raise RuntimeError("Can't read new CSV file to NodeCut that was already locked by the \"prepare\" method")
		# Read the CSV
		nodeDf=pd.read_csv(fileName)
		# Remove the gate column
		# df=df.drop(['gate'], axis=1)
		# Normalize labels
		if self.__nodeEmbedDf is None:
			self.__nodeEmbedDf = {cktId : nodeDf}
		else:
			self.__nodeEmbedDf[cktId] = nodeDf

	## Access the Pandas Dataframe
	#
	# \param self Instance of PathDataset class.
	# \return Pandas Dataframe object
	@property
	def df(self):
		return self.__df

	## Access the Pandas Dataframe with node embeddings
	#
	# \param self Instance of PathDataset class.
	# \return Pandas Dataframe object
	@property
	def nodeEmbedDf(self):
		return self.__nodeEmbedDf

	## Access object type
	#
	# \param self Instance of PathDataset class.
	# \return Bool identifying if object is for training or not
	@property
	def train(self):
		return self.__train

	## Normalizes all features
	#
	# Private method to normalize features when preparing
	#
	# \param self
	def __normalize(self):
		if self.__lock:
			raise RuntimeError("Already prepared")
		# Normalize labels
		self.__maxNumFanout=self.__df[[NodeCut.lNumFanout]].max()
		self.__df[[NodeCut.lNumFanout]]=(self.__df[[NodeCut.lNumFanout]]-self.__df[[NodeCut.lNumFanout]].min())/(self.__df[[NodeCut.lNumFanout]].max()-self.__df[[NodeCut.lNumFanout]].min())
		self.__maxNodeLevel=self.__df[[NodeCut.lNodeLevel]].max()
		self.__df[[NodeCut.lNodeLevel]]=(self.__df[[NodeCut.lNodeLevel]]-self.__df[[NodeCut.lNodeLevel]].min())/(self.__df[[NodeCut.lNodeLevel]].max()-self.__df[[NodeCut.lNodeLevel]].min())
		# self.__maxNodeRelativeLevel=self.__df[[NodeCut.lNodeRelativeLvl]].max()
		# self.__df[[NodeCut.lNodeRelativeLvl]]=(self.__df[[NodeCut.lNodeRelativeLvl]]-self.__df[[NodeCut.lNodeRelativeLvl]].min())/(self.__df[[NodeCut.lNodeRelativeLvl]].max()-self.__df[[NodeCut.lNodeRelativeLvl]].min())
		self.__maxNodeHasInversion=self.__df[[NodeCut.lNodeHasInversion]].max()
		self.__df[[NodeCut.lNodeHasInversion]]=(self.__df[[NodeCut.lNodeHasInversion]]-self.__df[[NodeCut.lNodeHasInversion]].min())/(self.__df[[NodeCut.lNodeHasInversion]].max()-self.__df[[NodeCut.lNodeHasInversion]].min())
		self.__maxChild1HasInversion=self.__df[[NodeCut.lChild1HasInversion]].max()
		self.__df[[NodeCut.lChild1HasInversion]]=(self.__df[[NodeCut.lChild1HasInversion]]-self.__df[[NodeCut.lChild1HasInversion]].min())/(self.__df[[NodeCut.lChild1HasInversion]].max()-self.__df[[NodeCut.lChild1HasInversion]].min())
		self.__maxChild1Level=self.__df[[NodeCut.lChild1Level]].max()
		self.__df[[NodeCut.lChild1Level]]=(self.__df[[NodeCut.lChild1Level]]-self.__df[[NodeCut.lChild1Level]].min())/(self.__df[[NodeCut.lChild1Level]].max()-self.__df[[NodeCut.lChild1Level]].min())
		self.__maxChild1Fo=self.__df[[NodeCut.lChild1Fo]].max()
		self.__df[[NodeCut.lChild1Fo]]=(self.__df[[NodeCut.lChild1Fo]]-self.__df[[NodeCut.lChild1Fo]].min())/(self.__df[[NodeCut.lChild1Fo]].max()-self.__df[[NodeCut.lChild1Fo]].min())
		self.__maxChild2HasInversion=self.__df[[NodeCut.lChild2HasInversion]].max()
		self.__df[[NodeCut.lChild2HasInversion]]=(self.__df[[NodeCut.lChild2HasInversion]]-self.__df[[NodeCut.lChild2HasInversion]].min())/(self.__df[[NodeCut.lChild2HasInversion]].max()-self.__df[[NodeCut.lChild2HasInversion]].min())
		self.__maxChild2Level=self.__df[[NodeCut.lChild2Level]].max()
		self.__df[[NodeCut.lChild2Level]]=(self.__df[[NodeCut.lChild2Level]]-self.__df[[NodeCut.lChild2Level]].min())/(self.__df[[NodeCut.lChild2Level]].max()-self.__df[[NodeCut.lChild2Level]].min())
		self.__maxChild2Fo=self.__df[[NodeCut.lChild2Fo]].max()
		self.__df[[NodeCut.lChild2Fo]]=(self.__df[[NodeCut.lChild2Fo]]-self.__df[[NodeCut.lChild2Fo]].min())/(self.__df[[NodeCut.lChild2Fo]].max()-self.__df[[NodeCut.lChild2Fo]].min())
		self.__maxCutTruthTable=self.__df[[NodeCut.lCutTruthTable]].max()
		self.__df[[NodeCut.lCutTruthTable]]=(self.__df[[NodeCut.lCutTruthTable]]-self.__df[[NodeCut.lCutTruthTable]].min())/(self.__df[[NodeCut.lCutTruthTable]].max()-self.__df[[NodeCut.lCutTruthTable]].min())
		self.__maxCutIsInverted=self.__df[[NodeCut.lCutIsInverted]].max()
		self.__df[[NodeCut.lCutIsInverted]]=(self.__df[[NodeCut.lCutIsInverted]]-self.__df[[NodeCut.lCutIsInverted]].min())/(self.__df[[NodeCut.lCutIsInverted]].max()-self.__df[[NodeCut.lCutIsInverted]].min())
		self.__maxCutNumLeaves=self.__df[[NodeCut.lCutNumLeaves]].max()
		self.__df[[NodeCut.lCutNumLeaves]]=(self.__df[[NodeCut.lCutNumLeaves]]-self.__df[[NodeCut.lCutNumLeaves]].min())/(self.__df[[NodeCut.lCutNumLeaves]].max()-self.__df[[NodeCut.lCutNumLeaves]].min())
		self.__maxCutVolume=self.__df[[NodeCut.lCutVolume]].max()
		self.__df[[NodeCut.lCutVolume]]=(self.__df[[NodeCut.lCutVolume]]-self.__df[[NodeCut.lCutVolume]].min())/(self.__df[[NodeCut.lCutVolume]].max()-self.__df[[NodeCut.lCutVolume]].min())
		self.__maxMinCutLvl=self.__df[[NodeCut.lCutMinLvl]].max()
		self.__df[[NodeCut.lCutMinLvl]]=(self.__df[[NodeCut.lCutMinLvl]]-self.__df[[NodeCut.lCutMinLvl]].min())/(self.__df[[NodeCut.lCutMinLvl]].max()-self.__df[[NodeCut.lCutMinLvl]].min())
		self.__maxMaxCutLvl=self.__df[[NodeCut.lCutMaxLvl]].max()
		self.__df[[NodeCut.lCutMaxLvl]]=(self.__df[[NodeCut.lCutMaxLvl]]-self.__df[[NodeCut.lCutMaxLvl]].min())/(self.__df[[NodeCut.lCutMaxLvl]].max()-self.__df[[NodeCut.lCutMaxLvl]].min())
		self.__maxCutLvl=self.__df[[NodeCut.lCutLvl]].max()
		self.__df[[NodeCut.lCutLvl]]=(self.__df[[NodeCut.lCutLvl]]-self.__df[[NodeCut.lCutLvl]].min())/(self.__df[[NodeCut.lCutLvl]].max()-self.__df[[NodeCut.lCutLvl]].min())
		self.__maxMinCutFo=self.__df[[NodeCut.lCutMinFo]].max()
		self.__df[[NodeCut.lCutMinFo]]=(self.__df[[NodeCut.lCutMinFo]]-self.__df[[NodeCut.lCutMinFo]].min())/(self.__df[[NodeCut.lCutMinFo]].max()-self.__df[[NodeCut.lCutMinFo]].min())
		self.__maxMaxCutFo=self.__df[[NodeCut.lCutMaxFo]].max()
		self.__df[[NodeCut.lCutMaxFo]]=(self.__df[[NodeCut.lCutMaxFo]]-self.__df[[NodeCut.lCutMaxFo]].min())/(self.__df[[NodeCut.lCutMaxFo]].max()-self.__df[[NodeCut.lCutMaxFo]].min())
		self.__maxCutFo=self.__df[[NodeCut.lCutFo]].max()
		self.__df[[NodeCut.lCutFo]]=(self.__df[[NodeCut.lCutFo]]-self.__df[[NodeCut.lCutFo]].min())/(self.__df[[NodeCut.lCutFo]].max()-self.__df[[NodeCut.lCutFo]].min())

	## Shuffles data set
	#
	# \param self
	def shuffle(self):
		self.__df=self.__df.sample(frac=1).reset_index(drop=True)

	## Prepares data set to be used
	#
	# Normalizes all features, shuffle data set and locks it so that no more
	# CSVs can be read.
	#
	# \param self
	# \param numTrainPoints is int defining # of training points to use
	# \param numValPoints is optional int defining # of validation points to use, -1 is default meaning all remaining data points
	# \param balanced is optional Bool to define if training set must have balanced representativity between label classes
	def prepare(self, numTrainPoints, numValPoints=-1, balanced=True):
		print("Preparing dataframe!")
		if self.__lock:
			raise RuntimeError("Already prepared")
		# self.__normalize()
		# # Shuffles data (NO NEED TO SHUFFLE WHEN USING sample)
		# self.shuffle()
		# Initializes all dataType columns to none or to validation
		if numValPoints<0:
			self.__df[NodeCut.lDataType]=NodeCut.lDataTypeVal
		else:
			self.__df[NodeCut.lDataType]=NodeCut.lDataTypeNone
		if numTrainPoints > 0:
			if balanced:
				classStep=1/self.__numClasses
				for classIdx in range(self.__numClasses):
					df=self.__df[(self.__df[NodeCut.lCutDelay]>=classStep*classIdx) &
					             (self.__df[NodeCut.lCutDelay]<=classStep*(classIdx+1))].sample(n=int(numTrainPoints/self.__numClasses))
					df[NodeCut.lDataType]=NodeCut.lDataTypeTrain
					self.__df.update(df)
			else:
				df=self.__df.sample(n=numTrainPoints)
				df[NodeCut.lDataType]=NodeCut.lDataTypeTrain
				self.__df.update(df)
		# Adds validation points if required
		if numValPoints>=0:
			df=self.__df[self.__df[NodeCut.lDataType]==NodeCut.lDataTypeNone].sample(n=numValPoints)
			df[NodeCut.lDataType]=NodeCut.lDataTypeVal
			self.__df.update(df)
		# print(len(self.__df.index))
		# print(len(self.__df[self.__df[NodeCut.lDataType]==NodeCut.lDataTypeTrain].index))
		# print(len(self.__df[self.__df[NodeCut.lDataType]==NodeCut.lDataTypeVal].index))
		# print(len(self.__df[self.__df[NodeCut.lDataType]==NodeCut.lDataTypeNone].index))
		# Set indexes of node embedding DF
		for embedKeys in self.__nodeEmbedDf:
			self.__nodeEmbedDf[embedKeys].set_index(NodeCut.lEmbedId, inplace=True)
		count_row = self.__df.shape[0]  # gives number of row count
		print("Initial shape = " + str(count_row))
		#self.__df.head()
		#df=self.__df.sort_values(by=[NodeCut.lCutDelay])
		#self.__df.head()
		#df=self.__df.drop_duplicates(subset=[NodeCut.lNumFanout,NodeCut.lNodeLevel,NodeCut.lNodeHasInversion,NodeCut.lChild1HasInversion,NodeCut.lChild1Level,NodeCut.lChild1Fo,NodeCut.lChild2HasInversion,NodeCut.lChild2Level,NodeCut.lChild2Fo,NodeCut.lCutIsInverted,NodeCut.lCutNumLeaves,NodeCut.lCutVolume,NodeCut.lCutMinLvl,NodeCut.lCutMaxLvl,NodeCut.lCutLvl,NodeCut.lCutMinFo,NodeCut.lCutMaxFo,NodeCut.lCutFo,NodeCut.lNodeRelativeLvl], keep='first')
		# Lock
		count_row = self.__df.shape[0]  # gives number of row count
		print("Final shape = " + str(count_row))

		self.__lock=True

	## Returns a tuple of Features and Labels
	#
	# Private method that creates features from a data frame and defines class
	# of labels. Returns values as tuple
	#
	# \param self
	# \param df is a Pandas Dataframe to be used for generating features and labels
	# \param nodeEmbedDict is a dictionary with DFs of node embeddings
	# \return tuple with list of features, list of labels, list of Node IDs and list of cut IDs
	def __getFeatureLabelTuple(self, df, nodeEmbedDict):
		if self.train:
			featureDF = df[[NodeCut.lCktId,NodeCut.lNodeId,NodeCut.lNumFanout,NodeCut.lNodeLevel,NodeCut.lNodeHasInversion,NodeCut.lChild1HasInversion,NodeCut.lChild1Level,NodeCut.lChild1Fo,NodeCut.lChild2HasInversion,NodeCut.lChild2Level,NodeCut.lChild2Fo,NodeCut.lCutTruthTable,NodeCut.lCutIsInverted,NodeCut.lCutNumLeaves,NodeCut.lCutVolume,NodeCut.lCutMinLvl,NodeCut.lCutMaxLvl,NodeCut.lCutLvl,NodeCut.lCutMinFo,NodeCut.lCutMaxFo,NodeCut.lCutFo,NodeCut.lCutIdx,NodeCut.l1id,NodeCut.l2id,NodeCut.l3id,NodeCut.l4id,NodeCut.l5id,NodeCut.lNodeRelativeLvl,NodeCut.lCutDelay]]
		else:
			featureDF = df[[NodeCut.lCktId,NodeCut.lNodeId,NodeCut.lNumFanout,NodeCut.lNodeLevel,NodeCut.lNodeHasInversion,NodeCut.lChild1HasInversion,NodeCut.lChild1Level,NodeCut.lChild2HasInversion,NodeCut.lChild1Fo,NodeCut.lChild2Level,NodeCut.lChild2Fo,NodeCut.lCutTruthTable,NodeCut.lCutIsInverted,NodeCut.lCutNumLeaves,NodeCut.lCutVolume,NodeCut.lCutMinLvl,NodeCut.lCutMaxLvl,NodeCut.lCutLvl,NodeCut.lCutMinFo,NodeCut.lCutMaxFo,NodeCut.lCutFo,NodeCut.lCutIdx,NodeCut.l1id,NodeCut.l2id,NodeCut.l3id,NodeCut.l4id,NodeCut.l5id,NodeCut.lNodeRelativeLvl]]
		# Initialize control variables
		classStep=1/self.__numClasses
		featureList=[]
		labelList=[]
		idList=[]
		idCutList=[]
		# Creates lists
		for index, row in featureDF.iterrows():
			feature=[]
			feature.append([row[NodeCut.lNumFanout],row[NodeCut.lNodeLevel],row[NodeCut.lNodeRelativeLvl],row[NodeCut.lNodeHasInversion],row[NodeCut.lChild1HasInversion],row[NodeCut.lChild1Level],row[NodeCut.lChild1Fo],row[NodeCut.lChild2HasInversion],row[NodeCut.lChild2Level],row[NodeCut.lChild2Fo]])
			#feature.append([row[NodeCut.lCutTruthTable],row[NodeCut.lCutTruthTable],row[NodeCut.lCutTruthTable],row[NodeCut.lCutTruthTable],row[NodeCut.lCutTruthTable],row[NodeCut.lCutTruthTable],row[NodeCut.lCutTruthTable],row[NodeCut.lCutTruthTable],row[NodeCut.lCutTruthTable]])
			# Gets ckt ID
			cktId = row[NodeCut.lCktId]
			# Loads index list of node Embedding DF
			indexList = nodeEmbedDict[cktId].index.values.tolist()
			for lLvl in [NodeCut.l1id, NodeCut.l2id, NodeCut.l3id, NodeCut.l4id, NodeCut.l5id]:
				if row[lLvl] in indexList:
					fo = nodeEmbedDict[cktId].loc[row[lLvl],NodeCut.lEmbedFo]
					# if not isinstance(fo, float):
					# 	raise TypeError("Parameter fo must be float. Value is: %s" % str(fo))
					lvl = nodeEmbedDict[cktId].loc[row[lLvl],NodeCut.lEmbedLvl]
					# if not isinstance(lvl, float):
					# 	raise TypeError("Parameter lvl must be float. Value is: %s" % str(lvl))
					inv = nodeEmbedDict[cktId].loc[row[lLvl],NodeCut.lEmbedInv]
					# if not isinstance(inv, float):
					# 	raise TypeError("Parameter inv must be float. Value is: %s" % str(inv))
					c1Inv = nodeEmbedDict[cktId].loc[row[lLvl],NodeCut.lEmbedC1Inv]
					# if not isinstance(c1Inv, float):
					# 	raise TypeError("Parameter c1Inv must be float. Value is: %s" % str(c1Inv))
					c1Lvl = nodeEmbedDict[cktId].loc[row[lLvl],NodeCut.lEmbedC1Lvl]
					# if not isinstance(c1Lvl, float):
					# 	raise TypeError("Parameter c1Lvl must be float. Value is: %s" % str(c1Lvl))
					c1Fo = nodeEmbedDict[cktId].loc[row[lLvl],NodeCut.lEmbedC1Fo]
					# if not isinstance(c1Fo, float):
					# 	raise TypeError("Parameter c1Fo must be float. Value is: %s" % str(c1Fo))
					c2Inv = nodeEmbedDict[cktId].loc[row[lLvl],NodeCut.lEmbedC2Inv]
					# if not isinstance(c2Inv, float):
					# 	raise TypeError("Parameter c2Inv must be float. Value is: %s" % str(c2Inv))
					c2Lvl = nodeEmbedDict[cktId].loc[row[lLvl],NodeCut.lEmbedC2Lvl]
					# if not isinstance(c2Lvl, float):
					# 	raise TypeError("Parameter c2Lvl must be float. Value is: %s" % str(c2Lvl))
					c2Fo = nodeEmbedDict[cktId].loc[row[lLvl],NodeCut.lEmbedC2Fo]
					# if not isinstance(c2Fo, float):
					# 	raise TypeError("Parameter c2Fo must be float. Value is: %s" % str(c2Fo))
					rlvl = nodeEmbedDict[cktId].loc[row[lLvl],NodeCut.lEmbedRLvl]
					# if not isinstance(rlvl, float):
					# 	raise TypeError("Parameter rlvl must be float. Value is: %s" % str(rlvl))
					feature.append([fo, lvl, inv, c1Inv, c1Lvl, c1Fo, c2Inv, c2Lvl, c2Fo, rlvl])
				else:
					feature.append([0,0,0,0,0,0,0,0,0,0])
			feature.append([row[NodeCut.lCutIsInverted],row[NodeCut.lCutIsInverted],row[NodeCut.lCutIsInverted],row[NodeCut.lCutIsInverted],row[NodeCut.lCutIsInverted],row[NodeCut.lCutIsInverted],row[NodeCut.lCutIsInverted],row[NodeCut.lCutIsInverted],row[NodeCut.lCutIsInverted],row[NodeCut.lCutIsInverted]])
			feature.append([row[NodeCut.lCutNumLeaves],row[NodeCut.lCutNumLeaves],row[NodeCut.lCutNumLeaves],row[NodeCut.lCutNumLeaves],row[NodeCut.lCutNumLeaves],row[NodeCut.lCutNumLeaves],row[NodeCut.lCutNumLeaves],row[NodeCut.lCutNumLeaves],row[NodeCut.lCutNumLeaves],row[NodeCut.lCutNumLeaves]])
			feature.append([row[NodeCut.lCutVolume],row[NodeCut.lCutVolume],row[NodeCut.lCutVolume],row[NodeCut.lCutVolume],row[NodeCut.lCutVolume],row[NodeCut.lCutVolume],row[NodeCut.lCutVolume],row[NodeCut.lCutVolume],row[NodeCut.lCutVolume],row[NodeCut.lCutVolume]])
			feature.append([row[NodeCut.lCutMinLvl],row[NodeCut.lCutMinLvl],row[NodeCut.lCutMinLvl],row[NodeCut.lCutMinLvl],row[NodeCut.lCutMinLvl],row[NodeCut.lCutMinLvl],row[NodeCut.lCutMinLvl],row[NodeCut.lCutMinLvl],row[NodeCut.lCutMinLvl],row[NodeCut.lCutMinLvl]])
			feature.append([row[NodeCut.lCutMaxLvl],row[NodeCut.lCutMaxLvl],row[NodeCut.lCutMaxLvl],row[NodeCut.lCutMaxLvl],row[NodeCut.lCutMaxLvl],row[NodeCut.lCutMaxLvl],row[NodeCut.lCutMaxLvl],row[NodeCut.lCutMaxLvl],row[NodeCut.lCutMaxLvl],row[NodeCut.lCutMaxLvl]])
			feature.append([row[NodeCut.lCutLvl],row[NodeCut.lCutLvl],row[NodeCut.lCutLvl],row[NodeCut.lCutLvl],row[NodeCut.lCutLvl],row[NodeCut.lCutLvl],row[NodeCut.lCutLvl],row[NodeCut.lCutLvl],row[NodeCut.lCutLvl],row[NodeCut.lCutLvl]])
			feature.append([row[NodeCut.lCutMinFo],row[NodeCut.lCutMinFo],row[NodeCut.lCutMinFo],row[NodeCut.lCutMinFo],row[NodeCut.lCutMinFo],row[NodeCut.lCutMinFo],row[NodeCut.lCutMinFo],row[NodeCut.lCutMinFo],row[NodeCut.lCutMinFo],row[NodeCut.lCutMinFo]])
			feature.append([row[NodeCut.lCutMaxFo],row[NodeCut.lCutMaxFo],row[NodeCut.lCutMaxFo],row[NodeCut.lCutMaxFo],row[NodeCut.lCutMaxFo],row[NodeCut.lCutMaxFo],row[NodeCut.lCutMaxFo],row[NodeCut.lCutMaxFo],row[NodeCut.lCutMaxFo],row[NodeCut.lCutMaxFo]])
			feature.append([row[NodeCut.lCutFo],row[NodeCut.lCutFo],row[NodeCut.lCutFo],row[NodeCut.lCutFo],row[NodeCut.lCutFo],row[NodeCut.lCutFo],row[NodeCut.lCutFo],row[NodeCut.lCutFo],row[NodeCut.lCutFo],row[NodeCut.lCutFo]])
			featureList.append(feature)
			if self.train:
				label=min(self.__numClasses-1,int(row[NodeCut.lCutDelay]/classStep))
				labelList.append(label)
			idList.append(row[NodeCut.lNodeId])
			idCutList.append(row[NodeCut.lCutIdx])
		# Returns tuple
		return featureList, labelList, idList, idCutList

	## Returns the shape of features
	#
	# \param self
	# \return list with shape of features
	def getFeatureShape(self):
		return [15,10,1]

	## Reshapes feature to use on TF
	#
	# \param self
	# \param npArray is Numpy array feature
	# \return npArray with reshaped feature
	def reshapeFeature(self, npArray):
		return np.reshape(npArray, [len(npArray)]+self.getFeatureShape())

	## Returns a tuple of Features and Labels for training
	#
	# \param self
	# \return tuple with list of features, list of labels, list of Node IDs and list of cut IDs
	def getTrainFeatureLabelTuple(self):
		if not self.__lock:
			raise RuntimeError("Object must first be prepared")
		return self.__getFeatureLabelTuple(self.__df[self.__df[NodeCut.lDataType]==NodeCut.lDataTypeTrain], self.__nodeEmbedDf)

	## Returns a tuple of Features and Labels for validation
	#
	# \param self
	# \return tuple with list of features, list of labels, list of Node IDs and list of cut IDs
	def getValFeatureLabelTuple(self):
		if not self.__lock:
			raise RuntimeError("Object must first be prepared")
		return self.__getFeatureLabelTuple(self.__df[self.__df[NodeCut.lDataType]==NodeCut.lDataTypeVal], self.__nodeEmbedDf)

	## Plots correlation between available features and labels
	#
	# \param self
	def plotTrainCorr(self):
		# Bins labels (values are always between 0 and 1)
		classBins=np.linspace(0, 1, self.__numClasses+1)
		for featLabel in [NodeCut.lNumFanout, NodeCut.lNodeLevel, NodeCut.lNodeHasInversion, NodeCut.lChild1HasInversion, NodeCut.lChild1Level, NodeCut.lChild2HasInversion, NodeCut.lChild2Level, NodeCut.lCutIsInverted, NodeCut.lCutNumLeaves, NodeCut.lCutVolume, NodeCut.lCutMinLvl, NodeCut.lCutMaxLvl, NodeCut.lCutLvl, NodeCut.lCutMinFo, NodeCut.lCutMaxFo, NodeCut.lCutFo]:
			# Collect binned data for NumFanout and generate yList to plot for each label class
			xList=[]
			yList=[]
			sizeList=[]
			for classIdx in range(self.__numClasses):
				# Include 1 for last iteration
				if classIdx+1==1:
					df=self.__df[(self.__df[NodeCut.lDataType]==NodeCut.lDataTypeTrain) & (self.__df[NodeCut.lCutDelay]>=classBins[classIdx]) & (self.__df[NodeCut.lCutDelay]<=classBins[classIdx+1])]
				else:
					df=self.__df[(self.__df[NodeCut.lDataType]==NodeCut.lDataTypeTrain) & (self.__df[NodeCut.lCutDelay]>=classBins[classIdx]) & (self.__df[NodeCut.lCutDelay]<classBins[classIdx+1])]
				df=df[featLabel].astype(float)
				dfCut, binLabelList=pd.qcut(df,duplicates="drop",q=10,retbins=True)
				binLabelList=binLabelList[1:]
				binSizeList=dfCut.value_counts().tolist()
				for binLabel, binSize in zip(binLabelList,binSizeList):
					xList.append(classIdx)
					yList.append(binLabel)
					sizeList.append(binSize)
			fig = px.scatter(x=xList,y=yList,size=sizeList,title="%s (y) vs CutDelay (x)" % featLabel)
			fig.show()
