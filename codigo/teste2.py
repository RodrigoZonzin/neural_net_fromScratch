import numpy as np

W = np.random.rand(7)  
bias = np.random.rand() 

def psi(z): 
    return 1 if z >= 0 else -1

def predict(X):
    return psi(np.dot(X, W) + bias)

def erro(y_hat, y):
    return y_hat - y

k = 0.1
epocas = 10

X_train = np.load('xtrain.npy')[:1000]
Y_train = np.load('ytrain.npy')[:1000]

X_test = np.load('xtrain.npy')[1000:]
Y_test = np.load('ytrain.npy')[1000:]

print(W)
for i in range(epocas):
    print(f"Época: {i}")
    for xl, fxl in zip(X_train, Y_train):
        var_erro = erro(predict(xl), fxl)
        print(f"Erro: {var_erro}")
        dW = -k * var_erro * xl
        W = W + dW
        bias = bias + (-k * var_erro)  # Atualiza o bias
        print(f"Pesos: {W}")
        print(f"Bias: {bias}")
        print("--"*20)


corretos_train = 0
for i, Xi in enumerate(X_train):
    y_train = predict(Xi)

    if y_train == Y_train[i]:
        corretos_train += 1

acuracia_train = corretos_train / len(X_train)


corretos = 0
for i, Xi in enumerate(X_test):
    y = predict(Xi)

    if y == Y_test[i]:
        #print(y, Y_test[i])
        corretos += 1

print(f"Acurácia de Treino: {corretos_train/len(X_train)}")
print(f"Acurácia de Teste: {corretos/len(X_test)}")
    