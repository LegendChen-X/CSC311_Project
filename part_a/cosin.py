import math
import numpy as np
from utils import *

def cosineSimilar(matrix, k, i, j, cosine):
    neighbour = -np.ones((k, 2), dtype="float")
    flag = 10
    for y in range(matrix.shape[0]):
        if y == i: continue
        if not np.isnan(matrix[i][j]) and cosine[i][y]:
            cos = cosine[i][y]
            minIndex = 0
            minVal = 999999999999999
            for z in range(k):
                if neighbour[z][1] < minVal:
                    minVal = neighbour[z][1]
                    minIndex = z
            neighbour[minIndex][0] = y
            neighbour[minIndex][1] = cos
    count = 0.001
    missing_val = 0
    for z in range(k):
        if neighbour[z][0] == -1: continue
        missing_val += matrix[int(neighbour[z][0])][j]
        count += 1
    return missing_val / count

def knn_impute_by_user(matrix, valid_data, k):
    mat = matrix.copy()
    height, width = mat.shape
    upperMat = np.zeros((mat.shape[0], mat.shape[0]), dtype="float")
    for i in range(matrix.shape[0]):
        for k in range(matrix.shape[0]):
            if i == k: continue
            if upperMat[k][i]:
                upperMat[i][k] = upperMat[k][i]
                continue
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i][j]) and not np.isnan(matrix[k][j]): upperMat[i][k] += matrix[i][j] * matrix[k][j]
                
    norms = np.zeros((mat.shape[0],), dtype="float")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i][j]): norms[i] += matrix[i][j] ** 2
    norms = np.sqrt(norms)
    
    cosine = np.zeros((mat.shape[0], mat.shape[0]), dtype="float")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i == k: continue
            cosine[i][j] = upperMat[i][k] / (norms[i] * norms[j])
    
    for i in range(height):
        for j in range(width):
            if np.isnan(mat[i][j]):
                mat[i][j] = cosineSimilar(mat, k, i, j, cosine)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy User-based with k = {} : {}".format(k, acc))
    return acc
    
def sparse_matrix_evaluate(data, matrix, threshold=0.5):
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_user_id, cur_question_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)

def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    k_set = [1, 6, 11, 16, 21, 26]
    accuracy_user = []
    accuracy_item = []
    for k in k_set:
        accuracy_user.append(knn_impute_by_user(sparse_matrix,val_data,k))
        best_k_user = k_set[accuracy_user.index(max(accuracy_user))]
    test_user = knn_impute_by_user(sparse_matrix,test_data,best_k_user)
    print("Best k for user is", best_k_user, "with test acc:", test_user)

if __name__ == "__main__":
    main()
