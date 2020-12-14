# TODO: complete this file.
from knn import *
from item_response import *
import numpy as np
import torch

def sample(data):
    indexes = np.random.randint(len(data["user_id"]), size=len(data["user_id"]))
    U = np.array(data["user_id"])[indexes].tolist()
    Q = np.array(data["question_id"])[indexes].tolist()
    C = np.array(data["is_correct"])[indexes].tolist()
    sample = {"user_id": U, "question_id": Q, "is_correct":C}
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

def bag_accuracy(data, predictions, m):
    prob = np.zeros((len(data["is_correct"]),), dtype="float")
    predict = np.zeros((len(data["is_correct"]),), dtype="float")
    for i in range(len(data["is_correct"])):
        total = 0
        for j in range(m): total += predictions[j][i]
        prob[i] = total / m
        if prob[i] < 0.5: predict[i] = 0
        else: predict[i] = 1
    return np.sum((data["is_correct"] == predict)) / len(data["is_correct"])    
    
def bag(trainData, validData, testData, m):
    res_val = []
    res_test = []
    for i in range(m):
        print("####Sample {}".format(i+1))
        sampleData = sample(trainData)
        best_theta, best_beta, val_acc_lst, train_neg_lld_lst, valid_neg_lld_lst = irt(sampleData, validData, 0.002, 130)
        res_val.append(newEvaluate(validData, best_theta, best_beta))
        res_test.append(newEvaluate(testData, best_theta, best_beta))
    
    valid_acc = bag_accuracy(validData, res_val, m)
    test_acc = bag_accuracy(testData, res_test, m)
   
    return valid_acc, test_acc
       
def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    valid_acc, test_acc = bag(train_data, val_data, test_data, 3)
    print("Final validation accuracy is {}".format(valid_acc))
    print("Final test accuracy is {}".format(test_acc))

if __name__ == "__main__":
    main()
    
    
