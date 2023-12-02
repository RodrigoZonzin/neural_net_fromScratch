import random 
import numpy as np

X_train = []
Y_train = []
for i in range(1500):
    X = [random.randint(0, 9) for i in range(7)]

    value = int(''.join(map(str, X)))
    if value < 4646407:
        Y = [-1]
    else:
        Y = [1]
    #print(f"[{X}, {Y}],")
    X_train.append(X)
    Y_train.append(Y)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

np.save("xtrain.npy", X_train)
np.save("ytrain.npy", Y_train)

#print(Y_train)