#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: li
'''

import pickle
import numpy
import heapq
import bottleneck


def train(x,y):
    for i in x:
        print(i)


def compute_distances(x):
    for i in x:
        print(i)


bottleneck
train_data = pickle.load(open("/Users/wenzhongli/PycharmProjects/CS536_P1/cs536_1/models/train.pkl", "rb" ))
test_data = pickle.load(open("/Users/wenzhongli/PycharmProjects/CS536_P1/cs536_1/models/test.pkl","rb"))
val_ratio = 0.2

categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
heapq.nsmallest()
#student_parameters
#You may want to change these in your experiment later.
train_ratio= 1.0
train_num = int(train_data['data'].shape[0]*train_ratio*(1.0-val_ratio))
val_num = -1*int(train_data['data'].shape[0]*train_ratio*val_ratio)

# train(train_data['data'][:train_num], train_data['target'][:train_num])
compute_distances(train_data['data'][val_num:,:])
count = 0
for line in train_data:
    print(str(line))
    count += 1
    if count == 100:
        break












#
# For K= 1 and train_ratio= 0.200000, Got 1049 / 1353 correct => VAL_accuracy: 0.775314
# For K= 3 and train_ratio= 0.200000, Got 1063 / 1353 correct => VAL_accuracy: 0.785661
# For K= 5 and train_ratio= 0.200000, Got 1081 / 1353 correct => VAL_accuracy: 0.798965
# For K= 8 and train_ratio= 0.200000, Got 1102 / 1353 correct => VAL_accuracy: 0.814486
# For K= 10 and train_ratio= 0.200000, Got 1111 / 1353 correct => VAL_accuracy: 0.821138
# For K= 12 and train_ratio= 0.200000, Got 1108 / 1353 correct => VAL_accuracy: 0.818921
# For K= 15 and train_ratio= 0.200000, Got 1118 / 1353 correct => VAL_accuracy: 0.826312
# For K= 20 and train_ratio= 0.200000, Got 1113 / 1353 correct => VAL_accuracy: 0.822616
# For K= 50 and train_ratio= 0.200000, Got 1087 / 1353 correct => VAL_accuracy: 0.803400
# For K= 100 and train_ratio= 0.200000, Got 1064 / 1353 correct => VAL_accuracy: 0.786401
