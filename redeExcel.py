import numpy as np

class MyArray(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __str__(self):
        formatted_str = ','.join(map(str, self.flatten()))
        return formatted_str

W = MyArray(np.random.rand(7))
bias = (np.random.rand())

def phi(z): 
    return 1 if z >= 0 else -1

def predict(X):
    return phi(np.dot(X, W) + bias)

def erro(y_hat, y):
    return y_hat - y

k = 0.1
epocas = 10

X_train = MyArray(np.load('xtrain.npy')[:1000])
Y_train = MyArray(np.load('ytrain.npy')[:1000])

X_test = MyArray(np.load('xtrain.npy')[1000:])
Y_test = MyArray(np.load('ytrain.npy')[1000:])

print(f"W inicial aleatorio, {W}")
print(f"Bias inicial aleatorio, {bias}")
print(","*8)
print(","*8)

iteracao = 0
for i in range(epocas):
    for xl, fxl in zip(X_train, Y_train):
        
        var_erro = erro(predict(xl), fxl)
        dW = -k * var_erro * xl
        # Atualiza o bias
        print(f"iteracao{iteracao}epoca{i}"+",-"*7)
        print(f"pesos, {W}")
        print(f"entrada, {xl}")
        print(f"pesos*entrada, {W*xl}")
        print(f"prodInterno(peso; entrada), {W.dot(xl)},,,,,")
        print(f"valor_obtido, {predict(xl)},valor_esperado,{fxl},,,")
        print(f"erro, {var_erro},,,,,")
        print(f"dW, {dW}")
        
        W = W + dW
        bias = bias + (-k * var_erro) 
        print(f"PesoAtualizado,{W}")
        print(f"NovoBias, {bias}")
        print("-,"*8)
        print(","*8)
        print(","*8)
        iteracao += 1


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
    