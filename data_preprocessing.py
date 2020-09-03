# libraries
import numpy as np
import os
import pandas as pd
import cv2

# constant variables
NUMBER_OF_TRAIN_IMAGES = 700
IMAGE_SIZE = 96
NEW_DIMENSION = (96, 96)

# paths
data_path = "/home/sansingh/github/flower_classification/raw_data/"
processed_data_path = "/home/sansingh/github/flower_classification/processed_data/" 

# function to get filename from a path
def get_filename(path):
	splits = path.split("/")
	filename = splits[len(splits) - 1]
	return filename

# collect paths of all images and their labels
train_df = pd.DataFrame()
test_df = pd.DataFrame()
dirs = os.listdir(data_path)
for i in range(len(dirs)):
	paths = []
	labels = []
	current_path = data_path + dirs[i] + "/"
	files = os.listdir(current_path)
	for j in range(len(files)):
		paths.append(current_path + files[j])
		labels.append(dirs[i])
	df = pd.DataFrame()
	df['paths'] = paths
	df['labels'] = labels
	df = df.sample(frac=1.0).reset_index(drop=True)
	train_df = train_df.append(df.iloc[0:NUMBER_OF_TRAIN_IMAGES, :], ignore_index=True)
	test_df = test_df.append(df.iloc[NUMBER_OF_TRAIN_IMAGES:, :], ignore_index=True)
	print(str(i + 1) + " of " + str(len(dirs)) + " done.")
train_df = train_df.sample(frac=1.0).reset_index(drop=True)
test_df = test_df.sample(frac=1.0).reset_index(drop=True)

# read images, resize and save np object
train_npy = np.ndarray((train_df.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i in range(train_df.shape[0]):
	img = cv2.imread(train_df['paths'][i])
	img = cv2.resize(img, NEW_DIMENSION)
	current_target_path = processed_data_path + train_df['labels'][i] + "/"
	if(os.path.exists(current_target_path) == False):
		os.makedirs(current_target_path)
	cv2.imwrite(current_target_path + get_filename(train_df['paths'][i]), img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	train_npy[i] = img
	if(i % 1000 == 0):
		print(str(i + 1) + " of " + str(train_npy.shape[0]) + " done")
test_npy = np.ndarray((test_df.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i in range(test_df.shape[0]):
	img = cv2.imread(test_df['paths'][i])
	img = cv2.resize(img, NEW_DIMENSION)
	current_target_path = processed_data_path + test_df['labels'][i] + "/"
	if(os.path.exists(current_target_path) == False):
		os.makedirs(current_target_path)
	cv2.imwrite(current_target_path + get_filename(test_df['paths'][i]), img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	
	test_npy[i] = img
	if(i % 1000 == 0):
		print(str(i + 1) + " of " + str(test_npy.shape[0]) + " done")

# save npy files
np.save(processed_data_path + "train.npy", train_npy)
np.save(processed_data_path + "test.npy", test_npy)
train_df.to_csv(processed_data_path + "train_labels.csv", index=False)
test_df.to_csv(processed_data_path + "test_labels.csv", index=False)
print("processed data saved successfully")
