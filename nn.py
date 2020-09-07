# libraries
import numpy as np
import tensorflow as tf

# constants
SEED = 89
IMAGE_SIZE = 96
IMAGE_CHANNEL = 3

# paths
train_npy_path = "/home/sansingh/github/flower_classification/processed_data/train_images.npy"
train_labels_npy_path = "/home/sansingh/github/flower_classification/processed_data/train_labels_onehot.npy"
val_npy_path = "/home/sansingh/github/flower_classification/processed_data/val_images.npy"
val_labels_npy_path = "/home/sansingh/github/flower_classification/processed_data/val_labels_onehot.npy"

# set random seed
tf.random.set_seed(SEED)
np.random.seed(SEED)

# read data
train_images = np.load(train_npy_path)
train_labels = np.load(train_labels_npy_path)
val_images = np.load(val_npy_path)
val_labels = np.load(val_labels_npy_path)

# get summary of data
train_label_ids = np.argmax(train_labels, axis=1)
val_label_ids = np.argmax(val_labels, axis=1)
unique_label_ids = np.unique(train_label_ids)

# update status of data
print("Train Images and Labels: ", train_images.shape, train_labels.shape)
print("Validation Images and Labels: ", val_images.shape, val_labels.shape)
print("Following is the classwise number of images in training data: ")
for i in range(len(unique_label_ids)):
	print("\t" + str(unique_label_ids[i]) + " class: " + str(len(train_label_ids[train_label_ids == i])))
print("Following is the classwise number of images in validation data: ")
for i in range(len(unique_label_ids)):
	print("\t" + str(unique_label_ids[i]) + " class: " + str(len(val_label_ids[val_label_ids == i])))

# defining neural network architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(IMAGE_SIZE, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL)))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(192, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(384, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
#model.add(tf.keras.layers.Conv2D(216, (3, 3), activation='relu'))
#model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.summary()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(384, activation='relu'))
model.add(tf.keras.layers.Dense(5))
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_images, train_label_ids, epochs=20, validation_data=(val_images, val_label_ids))
