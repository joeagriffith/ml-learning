import numpy as np
from sklearn import preprocessing

def scale_matrix(X, x_min, x_max, range_max, range_min):
    nom = (X-range_min) * (x_max-x_min)
    denom = range_max - range_min 
    denom = denom + (denom == 0)
    return x_min + nom/denom

def scale_matrix_mean(X):
    return X / np.mean(X)

x = np.array([
    [1, 2.2, 3, 4],
    [1, 2.2, 3, 6],
    [1, 2, 3, 8.5],
    [1, 2, 3, 4],
])

X = np.array([x, x, x])
X[0][0][2] = 3

maxi = X[0,:,:-1].max()
mini = X[0,:,:-1].min()

X_to_be_scaled = X[0,:,:-1]
# X_scaled = scale_matrix(X_to_be_scaled, 0, 1, maxi, mini)
X_scaled = X_to_be_scaled / np.mean(X_to_be_scaled)
print(X_scaled)

# Y_to_be_scaled = X[0,:,-1:]
# Y_scaled = scale_matrix(Y_to_be_scaled, 0, 1, maxi, mini)

# X[0] = np.append(X_scaled, Y_scaled, axis=1)
# print(X)

# y = [1.5, 2.1, 3]
# y[0] = scale_matrix(y[0], 0, 1, maxi, mini)
# print(y)