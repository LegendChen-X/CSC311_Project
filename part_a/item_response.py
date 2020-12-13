from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################       
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']
    log_cij = lambda x :  is_correct[x] * np.log(sigmoid(theta[user_id[x]] - beta[question_id[x]])) + (1 - is_correct[x]) * np.log(1 - sigmoid(theta[user_id[x]] - beta[question_id[x]]))
    log_lklihood = np.sum([log_cij(x) for x in range(len(is_correct))])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]
    for i in range(len(is_correct)):
        # Find the index for user and question
        u = user_id[i]
        q = question_id[i]
        # Update theta[u]
        sig = sigmoid(theta[u] - beta[q])
        dtheta = sig - is_correct[i]
        theta[u] -= lr*dtheta
        # Update beta[q]
        sig = sigmoid(theta[u] - beta[q])
        dbeta = is_correct[i] - sig
        beta[q] -= lr*dbeta
        
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


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
    # TODO: Initialize theta and beta.
    theta = np.random.rand(542)
    beta = np.random.rand(1774)

    val_acc_lst = []
    best_theta = None
    best_beta = None
    best_score = 0.0
    train_neg_lld_lst = []
    valid_neg_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_neg_lld_lst.append(neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        valid_neg_lld_lst.append(val_neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        if (score > best_score):
            best_score = score
            best_theta = theta.copy()
            best_beta = beta.copy()
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
        
    # TODO: You may change the return values to achieve what you want.
    return best_theta, best_beta, val_acc_lst, train_neg_lld_lst, valid_neg_lld_lst


def evaluate(data, theta, beta):
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
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # Set the hyper parameters
    lr = 0.002
    iterations = 160
    np.random.seed(299)
    theta, beta, val_acc_lst, train_neg_lld_lst, valid_neg_lld_lst = irt(train_data, val_data, lr, iterations)
    x_axis = [*range(iterations)]
    
    plt.figure(1)
    plt.title('Negative Log-likelihood')
    plt.plot(x_axis, train_neg_lld_lst, color='green', label='training')
    plt.plot(x_axis, valid_neg_lld_lst, color='blue', label='validation')
    plt.legend()
    plt.xlabel('iteration times')
    plt.ylabel('NLLK')
    plt.show()    
    
    train_lld_lst = [-x for x in train_neg_lld_lst]
    valid_lld_lst = [-x for x in valid_neg_lld_lst]
    plt.figure(2)
    plt.title('Log-likelihood')
    plt.plot(x_axis, train_lld_lst, color='green', label='training')
    plt.plot(x_axis, valid_lld_lst, color='blue', label='validation')
    plt.legend()
    plt.xlabel('iteration times')
    plt.ylabel('LLK')
    plt.show()       
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    valid_acc = evaluate(data=val_data, theta=theta, beta=beta)
    print("Final validation accuracy is {}".format(valid_acc))
    test_acc = evaluate(data=test_data, theta=theta, beta=beta)
    print("Final test accuracy is {}".format(test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
