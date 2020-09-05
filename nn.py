# libraries
import numpy as np
import tensorflow as tf

# constants
SEED = 89

# paths
train_npy_path = "/home/sansingh/github/flower_classification/processed_data/train_images.npy"
train_labels_npy_path = "/home/sansingh/github/flower_classification/processed_data/train_labels_onehot.npy"
val_npy_path = "/home/sansingh/github/flower_classification/processed_data/val_images.npy"
val_labels_npy_path = "/home/sansingh/github/flower_classification/processed_data/val_labesl_onehot.npy"

# set random seed
tf.random.set_seed(SEED)
np.random.seed(SEED)

# read data
train_images = np.load(train_npy_path)
train_labels = np.load(train_labels_npy_path)
val_images = np.load(val_npy_path)
val_labels = np.load(val_labels_npy_path)

# update status of data
print("Train Images and Labels: ", train_images.shape, train_labels.shape)
print("Validation Images and Labels: ", val_images.shape, val_labels.shape)


