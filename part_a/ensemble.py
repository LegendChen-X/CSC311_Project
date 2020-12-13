# TODO: complete this file.
from knn import *
from item_response import *
from neural_network import *
import numpy as np
import torch

def bootstrap():
    

def knn(k):
    sparse_matrix = load_train_sparse("../data").toarray()
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(sparse_matrix)
    valid_data = load_valid_csv("../data")
    for i in len(valid_data["is_correct"])
    
def ir():
    
def nn():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    k = 50
    lr = 0.05
    num_epoch = 10
    lamb = 0.01
    model = AutoEncoder(zero_train_matrix.shape[1], k)
    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, 0)
    
    

def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    
    
