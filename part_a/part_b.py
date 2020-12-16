from utils import *

import numpy as np
import matplotlib.pyplot as plt
import random

from test import *


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha):
    """ Compute the negative log-likelihood.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """    
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']
    log_cij = lambda x :  is_correct[x] * np.log(sigmoid(alpha[question_id[x]] * (theta[user_id[x]] - beta[question_id[x]]))) +\
        (1 - is_correct[x]) * np.log(1 - sigmoid(alpha[question_id[x]] * (theta[user_id[x]] - beta[question_id[x]])))
    log_lklihood = np.sum([log_cij(x) for x in range(len(is_correct))])

    return -log_lklihood


def update_theta_beta_alpha(data, lr, theta, beta, alpha):
    """ Update theta and beta using gradient descent.

        theta <- new_theta
        beta <- new_beta
        alpha <- new_alpha

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: tuple of vectors
    """
    
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]
    for i in range(len(is_correct)):
        # Find the index for user and question
        u = user_id[i]
        q = question_id[i]
        # Update alpha[q]
        sig = sigmoid(alpha[q] * (theta[u] - beta[q]))
        dalpha = (sig - is_correct[i])* (theta[u] - beta[q])     
        if ( alpha[q] - lr*dalpha > 0):
            alpha[q] -= lr*dalpha
        # Update theta[u]
        sig = sigmoid(alpha[q] * (theta[u] - beta[q]))
        dtheta = (sig - is_correct[i]) * alpha[q]
        theta[u] -= lr*dtheta
        # Update beta[q]
        sig = sigmoid(alpha[q] * (theta[u] - beta[q]))
        dbeta = (is_correct[i] - sig) * alpha[q]
        beta[q] -= lr*dbeta
        
    return theta, beta, alpha

def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """

    theta = np.random.rand(542)
    beta = np.random.rand(1774)
    alpha = np.ones(1774)
    
    val_acc_lst = []
    best_theta = None
    best_beta = None
    best_alpha = None
    best_score = 0.0
    train_neg_lld_lst = []
    valid_neg_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha)
        train_neg_lld_lst.append(neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta, alpha = alpha)
        valid_neg_lld_lst.append(val_neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta, alpha = alpha)[0]
        if (score > best_score):
            best_score = score
            best_theta = theta.copy()
            best_beta = beta.copy()
            best_alpha = alpha.copy()
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, alpha = update_theta_beta_alpha(data, lr, theta, beta, alpha)
        
    return best_theta, best_beta, best_alpha, val_acc_lst, train_neg_lld_lst, valid_neg_lld_lst

def evaluate(data, theta, beta, alpha):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"]), pred

def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Set the hyper parameters
    lr = 0.003
    iterations = 160
    np.random.seed(299)
    theta, beta, alpha, val_acc_lst, train_neg_lld_lst, valid_neg_lld_lst = irt(train_data, val_data, lr, iterations)
    print(alpha)
    
    test_theta, test_beta, test_val_acc_lst, test_train_neg_lld_lst, test_valid_neg_lld_lst = test_irt(train_data, val_data, lr, iterations)
    
    valid_acc, wrong_list = test_evaluate(data=val_data, theta=test_theta, beta=test_beta)
    valid_acc, pred = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)
    
    count = 0
    print("The total number of wrong predictions for the original model is", len(wrong_list))
    for i in range(len(wrong_list)):
        if pred[wrong_list[i]] == val_data["is_correct"][wrong_list[i]]:
            count += 1
    print("The new model get", count)
    
    x_axis = [*range(iterations)]
    
    plt.figure(1)
    plt.title('Negative Log-likelihood')
    plt.plot(x_axis, train_neg_lld_lst, color='green', label='training')
    plt.plot(x_axis, valid_neg_lld_lst, color='blue', label='validation')
    plt.legend()
    plt.xlabel('iteration times')
    plt.ylabel('NLLK')
    plt.show()    
     
    print("Final validation accuracy is {}".format(valid_acc))
    test_acc = evaluate(data=test_data, theta=theta, beta=beta, alpha=alpha)[0]
    print("Final test accuracy is {}".format(test_acc))
    
    ## Implement part(d)
    #q_list = random.sample(range(1,1774), 5)
    #color_list = ['b', 'g', 'r', 'c', 'm']
    
    #for i in range(5):
        #q = q_list[i]
        #c = color_list[i]
        #beta_j = beta[q]
        #theta_sort = np.sort(theta)
        #p_correct = sigmoid(theta_sort - beta_j)
        #plt.plot(theta_sort, p_correct, color = c, label = 'j{} {}'.format(i, q))
    #plt.xlim(-1.5, 4)
    #plt.xlabel("theta")
    #plt.ylabel("probability of the correct response")
    #plt.legend()
    #plt.show()

if __name__ == "__main__":
    main()
