import pandas as pd
import numpy as np
import random as rd
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity

def convert_data_to_R_test(data):
    m = max(set(data["Student_number"]))
    n = max(set(data["Course_number"]))
    R_test = np.zeros((m+1, n+1))
    for k in data.index:
        i, j, score = data.loc[k][["Student_number", "Course_number", "q12"]]
        R_test[i, j] = score
    R_test[R_test==0] = np.nan
    return R_test

def get_R_train(percentage):
    m, n = R_test.shape
    R_train = R_test.copy()
    l = []
    for i in range(m):
        for j in range(n):
            if not np.isnan(R_test[i, j]):
                l.append((i, j))
    l = rd.sample(l, int(len(l) * (1 - percentage)))
    
    for (i, j) in l:
        R_train[i, j] = np.nan
    return R_train

def calculate_bu_bi_mean():
    mean = np.nanmean(R_train)
    m, n = R_train.shape
    c = []
    for j in range(n):
        for i in range(m):
            if not np.isnan(R_train[i][j]):
                c.append(R_train[i][j] - mean)
    c = np.array(c)

    A = np.zeros((len(c), m + n))
    count = 0
    for j in range(n):
        for i in range(m):
            if not np.isnan(R_train[i][j]):
                A[count][i] = 1
                A[count][m + j] = 1
                count += 1
                
    bu = sp.sparse.linalg.lsmr(A, c, damp=1)[0][0:m]
    bi = sp.sparse.linalg.lsmr(A, c, damp=1)[0][m:]
    
    return bu, bi, mean

def get_R_tilde():
    m, n = R_train.shape
    R_tilde = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if not np.isnan(R_train[i][j]):
                R_tilde[i][j] = R_train[i][j] - mean - bu[i] - bi[j]
    R_tilde[R_tilde==0] = np.nan
    return R_tilde

def calculate_student_similarities():
    R_tilde_sparse = sp.sparse.csr_matrix(np.nan_to_num(R_tilde))
    return cosine_similarity(R_tilde_sparse, dense_output=False)

def get_R_hat():
    m, n = R_tilde.shape
    R_hat = np.zeros((m, n))
    for i in range(m):
        neighbors = student_similarities[i].nonzero()[1]
        neighbors = sorted(neighbors, key=lambda k: student_similarities[i, k])[:10]
        for j in range(n):
            R_hat[i][j] += mean + bu[i] + bi[j]
            a = 0
            b = 0
            for k in neighbors:
                if not np.isnan(R_tilde[k][j]):
                    a += student_similarities[i, k] * R_tilde[k][j]
                    b += abs(k)
            if a != 0 and b != 0:
                R_hat[i][j] += a / b
                if R_hat[i][j] <= 1:
                    R_hat[i][j] = 1 
                if R_hat[i][j] >= 5:
                    R_hat[i][j] = 5
    return R_hat

def train_error():
    m, n = R_train.shape
    error = []
    for i in range(m):
        for j in range(n):
            if not np.isnan(R_train[i][j]):
                error.append((R_train[i][j] - R_hat[i][j]) ** 2)
    return np.sqrt(np.mean(error))

def test_error():
    m, n = R_train.shape
    error = []
    for i in range(m):
        for j in range(n):
            if not np.isnan(R_test[i][j]) and np.isnan(R_train[i][j]):
                error.append((R_test[i][j] - R_hat[i][j]) ** 2)
    return np.sqrt(np.mean(error))

data = pd.read_csv("cse_anonymized.csv", sep='\t', header=None)
data.columns = [
    "Student_number", "Session", "Course_number", "Mark", "Grade",
    "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12"]

R_test = convert_data_to_R_test(data)
R_train = get_R_train(0.8)
bu, bi, mean = calculate_bu_bi_mean()
R_tilde = get_R_tilde()
student_similarities = calculate_student_similarities()
R_hat = get_R_hat()

print("Train error: ",train_error())
print("Test error: ", test_error())








