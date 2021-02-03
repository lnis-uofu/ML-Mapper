import tensorflow as tf

class CNN():

	## Constructor
	#
	# Creates object and model.
	#
	# \param self
	# \param featureShape is a list with shape of input features
	# \param numClasses is an integer with the number of classes to be classified
	# \return CNN model
	def __init__(self, featureShape, numClasses):
		# Clearup everything before running
		tf.keras.backend.clear_session()
		# Create model
		model = tf.keras.models.Sequential()
		# Add layers
		model.add(tf.keras.layers.Conv2D(128, (15,1), activation='relu', input_shape=featureShape))
		model.add(tf.keras.layers.Dropout(0.1))
		#model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 10)))
		# model.add(tf.keras.layers.Conv2D(128, (1,10), activation='relu'))
		# model.add(tf.keras.layers.Dropout(0.1))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(128, activation='relu'))
		model.add(tf.keras.layers.Dropout(0.1))
		model.add(tf.keras.layers.Dense(128, activation='relu'))
		model.add(tf.keras.layers.Dropout(0.1))
		model.add(tf.keras.layers.Dense(128, activation='relu'))
		model.add(tf.keras.layers.Dropout(0.1))
		model.add(tf.keras.layers.Dense(128, activation='relu'))
		model.add(tf.keras.layers.Dropout(0.1))
		model.add(tf.keras.layers.Dense(numClasses, activation='softmax'))
		# lossFunction = 'binary_crossentropy'
		# lossFunction = 'mean_squared_logarithmic_error'
		# lossFunction = 'mean_squared_error'
		# lossFunction = 'mean_absolute_error'
		lossFunction = 'sparse_categorical_crossentropy'
		model.compile(optimizer='adam', loss=lossFunction, metrics=["accuracy"])
		# Stores model
		self.__model = model

	## Access the model
	#
	# \param self Instance of PathDataset class.
	# \return Sequential TF model
	@property
	def model(self):
		return self.__model
