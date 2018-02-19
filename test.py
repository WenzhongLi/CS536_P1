# Train and Test data load
import pickle
import numpy as np

train_data = pickle.load(open("train.pkl", "rb"))

#Fixed_parameters
#Please do not change the fixed parameters

val_ratio  = 0.2

#student_parameters
#You may want to change these in your experiment later.
train_ratio= 1.0  # we split the train_data into 0.8:training

#Split Train data into Train and Validation (Ratio Train : Val = 4:1)
train_num = int(train_data['data'].shape[0]*train_ratio*(1.0-val_ratio))
val_num = -1*int(train_data['data'].shape[0]*train_ratio*val_ratio)

# Please implement Classifier in the file : k_nearest_neighbor.py
from cs536_1.models import KNearestNeighbor
KNN_classifier = KNearestNeighbor()
# KNN_classifier.train(train_data['data'][:train_num], train_data['target'][:train_num])
KNN_classifier.train(train_data['data'][:val_num], train_data['target'][:val_num])

# Please implement compute_distances() in the file : k_nearest_neighbor.py
dists=KNN_classifier.compute_distances(train_data['data'][:val_num])
# dists=KNN_classifier.compute_distances(train_data['data'][val_num:,:])

#For the different K, compute the validation accuracies.
#Please implement the predict_labels()
k_choices = [1, 3, 5, 7, 9, 11, 15, 19]
for k in k_choices:
    y_test_pred = KNN_classifier.predict_labels(dists,k)
    # num_correct = np.sum(y_test_pred == train_data['target'][val_num:])
    num_correct = np.sum(y_test_pred == train_data['target'][train_num:])
    accuracy = float(num_correct) / (-1*val_num)
    print('For K= %d and train_ratio= %f, Got %d / %d correct => VAL_accuracy: %f'
          % (k,train_ratio,num_correct,-1*val_num, accuracy))