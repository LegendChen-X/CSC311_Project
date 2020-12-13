# TODO: complete this file.
from knn import *
from item_response import *
from neural_network import *
import numpy as np
import torch

def sample(data):
    indexes = np.random.randint(len(data["user_id"]), size=len(data["user_id"]))
    sample = {"user_id":data["user_id"][indexes], "question_id":data["question_id"][indexes], "is_correct":data["is_correct"][indexes]}
    return sample

def sampleMat(matrix):
    indexes = np.random.randint(matrix.shape[0], size=matrix.shape[0])
    sampleMat = matrix[indexes, :]
    return sampleMat

def knn(sampleMat, validData):
    res = []
    nbrs = KNNImputer(n_neighbors=11)
    mat = nbrs.fit_transform(sampleMat)
    for i in len(validData["user_id"]):
        res.append()
    return res
    
def nn(sample, validData):
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    k = 50
    lr = 0.05
    num_epoch = 10
    lamb = 0.01
    model = AutoEncoder(sample.shape[1], k)
    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, 0)
    
def bag(trainData, validData):
    knn_res = knn(sample(trainData), validData)
    ir_res = ir(sample(trainData), validData)
    nn_res = nn(sample(trainData), validData)
    final_res = (knn_res + ir_res + nn_res) / 3
    predict = np.zereos((len(final_res), ), dtype="float")
    for i in range(len(final_res)):
        if i < 0.5: predict[i] = 0
        else: predict[i] = 1
    
def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    print(len(val_data["is_correct"]))
    print(len(val_data["user_id"]))
    print(len(valid_data["question_id"]))
    
    
