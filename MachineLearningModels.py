import numpy as np

def inverse_matrix(matrix):
    mat = np.array(matrix)
    det = np.linalg.det(mat)
    #Check if invertible
    if det == 0:
        return "The matrix has no inverse"

    return np.linalg.inv(mat)

def LinearRegression(X, y):
    #Need to add a column of ones at start to account for a constant
    ones = np.array([[1] for _ in range(len(X))])
    X = np.array(X)
    rows, cols = np.shape(X)
    X = np.append(ones, X, axis=1).reshape(rows, cols+1)
    y = np.array(y)

    #Calculating Theta using Linear Algebra
    XT = np.transpose(X)
    XTX = XT @ X
    XTX_inv = inverse_matrix(XTX)
    XTy = XT @ y
    theta = XTX_inv @ XTy
    return theta



def predict(theta, x):
    theta = np.array(theta)
    x = np.array(x)
    #Adding a one to the start to account for a constant
    x = np.append([1], x)
    return np.transpose(theta) @ x

