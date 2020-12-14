# TODO: complete this file.
from knn import *
from item_response import *
import numpy as np
import torch

def sample(data):
    indexes = np.random.randint(len(data["user_id"]), size=len(data["user_id"]))
    sample = {"user_id":data["user_id"][indexes], "question_id":data["question_id"][indexes], "is_correct":data["is_correct"][indexes]}
    return sample
    
def newEvaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a)
    return pred
    
def bag(trainData, validData, iter):
    res = []
    for i in range(iter):
        sampleData = sample(trainData)
        best_theta, best_beta, val_acc_lst, train_neg_lld_lst, valid_neg_lld_lst = irt(sampleData, validData, 0.002, 250)
        res.append(newEvaluate(validData, best_theta, best_beta))
    prob = np.zeros((len(validData["is_correct"]),), dtype="float")
    for i in range(validData["is_correct"]):
        total = 0
        for j in range(iter): total += res[j][i]
        prob[i] = total / iter
    predict = np.zeros((len(validData["is_correct"]),), dtype="float")
    for i in range(len(predict)):
        if prob[i] < 0.5: predict[i] = 0
        else: predict[i] = 1
    return np.sum((validData["is_correct"] == predict)) / len(validData["is_correct"])
    
def main():
    print("cxnb")
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    print(bag(train_data, val_data, 3))
    
if __name__ == "__main__":
    main()
    
    
