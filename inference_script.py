# libraries
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# constants
SEED = 89
IMAGE_SIZE = 96
IMAGE_CHANNEL = 3

# paths
val_npy_path = "/home/sansingh/github/flower_classification/processed_data/val_images.npy"
val_labels_npy_path = "/home/sansingh/github/flower_classification/processed_data/val_labels_onehot.npy"
test_npy_path = "/home/sansingh/github/flower_classification/processed_data/test_images.npy"
test_labels_npy_path = "/home/sansingh/github/flower_classification/processed_data/test_labels_onehot.npy"
model_target_path = "/home/sansingh/github/flower_classification/"
model_name = "saved_model_1"

# read data
val_images = np.load(val_npy_path)
val_labels = np.load(val_labels_npy_path)
test_images = np.load(test_npy_path)
test_labels = np.load(test_labels_npy_path)

# converting data type
val_images = val_images.astype(np.float32)
test_images = test_images.astype(np.float32)

# get summary of data
val_label_ids = np.argmax(val_labels, axis=1)
test_label_ids = np.argmax(test_labels, axis=1)

# summary of validation data
print("Validation Images and Labels: ", val_images.shape, val_labels.shape)
print("Test Images and Labels: ", test_images.shape, test_labels.shape)

# loading tflite model
interpreter = tf.lite.Interpreter(model_path = model_target_path + model_name + "/tflite_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# predicting for validation data
val_output_classes = []
for i in range(val_images.shape[0]):
	val_image = np.ndarray((1, 96, 96, 3), dtype=np.float32)
	val_image[0] = val_images[i]
	interpreter.set_tensor(input_details[0]['index'], val_image)
	interpreter.invoke()
	val_output_data = interpreter.get_tensor(output_details[0]['index'])
	val_output_classes.append(np.argmax(val_output_data[0], axis=0))

# evaluate results - validation data
val_accuracy = accuracy_score(val_output_classes, val_label_ids)
print("Validation Accuracy: ", val_accuracy * 100)
val_cfm = confusion_matrix(val_label_ids, val_output_classes)
print("Validation Confusion Matrix: ")
print(val_cfm)

# predicting for test data
test_output_classes = []
for i in range(test_images.shape[0]):
	test_image = np.ndarray((1, 96, 96, 3), dtype=np.float32)
	test_image[0] = test_images[i]
	interpreter.set_tensor(input_details[0]['index'], test_image)
	interpreter.invoke()
	test_output_data = interpreter.get_tensor(output_details[0]['index'])
	test_output_classes.append(np.argmax(test_output_data[0], axis=0))

# evaluate results - test data
test_accuracy = accuracy_score(test_output_classes, test_label_ids)
print("Test Accuracy: ", test_accuracy * 100)
test_cfm = confusion_matrix(test_label_ids, test_output_classes)
print("Test Confusion Matrix: ")
print(test_cfm)

