# libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

# constants
TEST_SIZE = 0.2
SEED = 89
IMAGE_SIZE = 96
IMAGE_CHANNELS = 3

# setting random seed
np.random.seed(SEED)

# paths
train_csv_path = "/home/sansingh/github/flower_classification/processed_data/train_labels.csv"
test_csv_path = "/home/sansingh/github/flower_classification/processed_data/test_labels.csv"
target_path = "/home/sansingh/github/flower_classification/processed_data/"
data_info_file = "/home/sansingh/github/flower_classification/processed_data/data_info.txt"

# open data_info_file in write mode
file_writer = open(data_info_file, "w")
file_writer.write("* * * This file has information about the processed dataset * * *\n")

# read data set
train_csv = pd.read_csv(train_csv_path)
test_csv = pd.read_csv(test_csv_path)

# make filenames
filenames = list(train_csv['paths'])
test_filenames = list(test_csv['paths'])

# make labels
unique_labels = list(train_csv['labels'].unique())
file_writer.write("Following are the indices of labels in the labels npy files:\n")
print("* * * Following are the indices of labels: * * *")
for i in range(len(unique_labels)):
	print(str(i) + ". " +  unique_labels[i])
	file_writer.write(str(i) + ". " +  unique_labels[i] + "\n")
print("")
all_labels = list(train_csv['labels'])
for i in range(len(all_labels)):
	label = all_labels[i]
	label_id = unique_labels.index(label)
	all_labels[i] = int(label_id)

# splitting data into train and validation
X_train, X_val, Y_train, Y_val = train_test_split(filenames, all_labels, test_size=TEST_SIZE, stratify=all_labels, random_state=SEED)
print("Train Shape: ", len(X_train), len(Y_train))
print("Validation Shape: ", len(X_val), len(Y_val))

# create ndarray
train_images = np.ndarray((len(X_train), IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
val_images = np.ndarray((len(X_val), IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
test_images = np.ndarray((len(test_filenames), IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))

# read images
for i in range(len(X_train)):
	img = cv2.imread(X_train[i])
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	train_images[i] = img
	if(i % 1000 == 0):
		print(str(i + 1) + " of " + str(len(X_train)) + " done")
for i in range(len(X_val)):
	img = cv2.imread(X_val[i])
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	val_images[i] = img
	if(i % 1000 == 0):
		print(str(i + 1) + " of " + str(len(X_train)) + " done")

# normalize images
train_images = train_images / 255.0
val_images = val_images / 255.0

# transform labels into one-hot vectors
Y_train_onehot = pd.get_dummies(Y_train)
Y_val_onehot = pd.get_dummies(Y_val)

# get label ids for test data
all_test_labels = list(test_csv['labels'])
for i in range(len(all_test_labels)):
	label = all_test_labels[i]
	label_id = unique_labels.index(label)
	all_test_labels[i] = int(label_id)
# read test images
for i in range(len(test_filenames)):
	img = cv2.imread(test_filenames[i])
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	test_images[i] = img
	if(i % 1000 == 0):
		print(str(i + 1) + " of " + str(len(test_filenames)) + " done")
#normalize test data
test_images = test_images / 255.0
# transform test labels into one-hot vectors
test_onehot = pd.get_dummies(all_test_labels)

# save all in npy files
print("Saving npys...")
file_writer.write("All npy files (except for labels) have normalized values - (divided by 255.0)\n")
file_writer.write("npy files for labels have one-hot values.\n")
np.save(target_path + "train_images.npy", train_images)
np.save(target_path + "val_images.npy", val_images)
np.save(target_path + "test_images.npy", test_images)
np.save(target_path + "train_labels_onehot.npy", np.array(Y_train_onehot))
np.save(target_path + "val_labels_onehot.npy", np.array(Y_val_onehot))
np.save(target_path + "test_labels_onehot.npy", np.array(test_onehot))
file_writer.close()
print("All npy files saved successfully...")
