import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_datasets as tfds
from datetime import datetime

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
	# Currently, memory growth needs to be the same across GPUs
	for gpu in gpus:
	  tf.config.experimental.set_memory_growth(gpu, True)
	logical_gpus = tf.config.list_logical_devices('GPU')
	print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
	# Memory growth must be set before GPUs have been initialized
	print(e)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


TRAIN_PERCENT = 0.8
VAL_PERCENT = 1.0 - TRAIN_PERCENT
LR = 0.001
EPOCH_NUM = 10
IMAGE_SIZE = (224, 224)

train_dir = 'data_img/imagenette2/train'
val_dir = 'data_img/imagenette2/val'
# Define some constants
BATCH_SIZE = 512
IMG_SIZE = (224, 224)

# Load the data
train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	train_dir,
	validation_split=0.2,
	subset='both',
	image_size=(320,320),
	batch_size=None,
	seed=123)


# Optional: Prepare dataset for performance
train_ds = train_ds.shuffle(1000)
def preprocess(image, label):
	image = tf.image.resize(image, (224,224))
	mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
	std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
	
	# Ensure the image is of type float32
	image = tf.cast(image, tf.float32) / 255.0
	
	# Normalize the image
	normalized_image = (image - mean) / std
	return normalized_image, label

# # Prefetch to improve performance
train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)
train_dataset = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

class SimpleCNN(Model):
	def __init__(self, num_classes=37):
		super(SimpleCNN, self).__init__()

		# Define layers
		self.conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')
		self.bn1 = layers.BatchNormalization()
		self.pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))

		self.conv3 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
		self.bn3 = layers.BatchNormalization()
		self.pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))

		self.conv5 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')
		self.bn5 = layers.BatchNormalization()
		self.pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))

		self.flatten = layers.Flatten()
		self.dense1 = layers.Dense(256)
		self.bn7 = layers.BatchNormalization()
		self.relu = layers.ReLU()

		self.output_layer = layers.Dense(num_classes, activation="softmax")

	def call(self, x, training=True):
		# First convolutional block
		x = self.conv1(x)
		x = self.bn1(x, training=training)
		x = self.relu(x)
		x = self.pool1(x)

		# Second convolutional block
		x = self.conv3(x)
		x = self.bn3(x, training=training)
		x = self.relu(x)
		x = self.pool2(x)

		# Third convolutional block
		x = self.conv5(x)
		x = self.bn5(x, training=training)
		x = self.relu(x)
		x = self.pool3(x)

		# Flatten and dense layers
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.bn7(x, training=training)
		x = self.relu(x)

		# Output layer
		x = self.output_layer(x)
		return x

new_model = SimpleCNN()

adam_optim = tf.keras.optimizers.Adam(learning_rate= LR)
new_model.compile(optimizer=adam_optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start_time = datetime.now()# Define data augmentation for the training dataset
new_model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCH_NUM)

end_time = datetime.now()
print("Whole training took:", (end_time - start_time).seconds)