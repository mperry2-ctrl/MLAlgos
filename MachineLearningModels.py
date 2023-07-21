import numpy as np
#Input all lists/matrices as an np.array

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
    rows, cols = np.shape(X)
    X = np.append(ones, X, axis=1).reshape(rows, cols+1)

    #Calculating Theta using Linear Algebra
    XT = np.transpose(X)
    XTX = XT @ X
    XTX_inv = inverse_matrix(XTX)
    XTy = XT @ y
    theta = XTX_inv @ XTy
    return theta

def GradientDescent(X, y, alpha, steps):
    #Need to add a column of ones at start to account for a constant
    ones = np.array([[1] for _ in range(len(X))])
    rows, cols = np.shape(X)
    X = np.append(ones, X, axis=1).reshape(rows, cols+1)

    #Initialize theta as a zero vector
    shape = (X.shape[1], 1)
    theta = np.zeros(shape)
    m = X.shape[0]
    for _ in range(steps):
        #Calculate Hypothesis Function output
        h = X @ theta
        #Diff between hypothesis and actual (calc of residual)
        difference = h - y
        #Calculating Gradient
        gradient = (np.transpose(X) @ difference) / m
        #Updating theta (alpha is also called the learning rate)
        theta -= alpha*gradient
    return theta


def predict(theta, x):
    #Adding a one to the start to account for a constant
    x = np.append([1], x)
    return np.transpose(theta) @ x



# Random data I can use for testing
# x = np.array([[0, 1], [2, 3], [3, 5], [4, 4], [10, 15]])
# y = np.array([[1], [2], [3], [6], [12]])
# print(GradientDescent(x, y, 0.001, 10000), LinearRegression(x, y))