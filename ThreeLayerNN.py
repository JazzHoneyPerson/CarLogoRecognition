# ThreeLayerNN.py
# Yu Fu
#
# Read features and output data;
# Divide data into training set and testing set in ratio of 9 : 1;
# Train a 3 layer neural network with node numbers (3200, 1600, 8)
# Save resulting weight paramenters into file. 
#
# Run for serveral times to generate different neural network,
# and use bootstrap aggregate in predicting.
#

import numpy as np
from scipy import optimize
from scipy import io
import Base

INPUT_LAYER_SIZE = Base.INPUT_LAYER_SIZE#количество нейронов на входном слое
HIDDEN_LAYER_SIZE = Base.HIDDEN_LAYER_SIZE#на внутреннем
OUTPUT_LAYER_SIZE = Base.OUTPUT_LAYER_SIZE#выходном

def sigmoid(z):
    return np.divide(1.0, 1.0 + np.exp(-1 * z))
#np.divide - делит один массив на другой


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def initialize_theta(l_in, l_out):
    eps_init = 0.12
    return np.random.rand(l_out, 1 + l_in) * 2 * eps_init - eps_init#инициализация весов происходит рандомным образом, 
#чтобы они были ближе к оптимальному варианту

'''
weight_1d: weight of each neuron #x
paras: (X, y, lambda)
'''
def costFunc(weights, *args):
    X, y, lam = args

    theta1 = np.reshape(weights[0:HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)],
                        (HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE + 1), order='F')
    theta2 = np.reshape(weights[HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1):],
                        (OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE + 1), order='F')
    m = np.size(X, 0)#высчитывем количество значений

    a1 = np.c_[np.ones((m, 1)), X]#добавляем вес активации?#input
    a2 = np.c_[np.ones((m, 1))'''Так же добавляем вес активации?''', sigmoid(a1.dot(np.transpose(theta1))'''Умножаем весы на значения''')'''Берем сигмоиду этого значения помноженную на весы''']#a.dot(b)-произведение матриц#hidden
    a3 = sigmoid(a2.dot(np.transpose(theta2)))#конечный результат#output

    yk = np.zeros((OUTPUT_LAYER_SIZE, m))#Функция zeros() создает массив из нулей
    for i in range(m):
        yk[y[i] - 1, i] = 1#создаем массив правильных ответов

    J = np.sum(np.sum(-1 * np.transpose(yk) * np.log(a3) - np.transpose(1 - yk) * np.log(1 - a3))) / m#сверяем ответы, я не понимаю что за херня здесь происходит

    theta1_unbiased = theta1[:, 1:]
    theta2_unbiased = theta2[:, 1:]
    reg = lam * (np.sum(np.sum(theta1_unbiased ** 2)) + np.sum(np.sum(theta2_unbiased ** 2))) / (2 * m)

    global itrCount
    print("Iteration {}: cost is {}".format(itrCount, J + reg))
    itrCount += 1

    return J + reg#возращаем количество ошибок

def gradFunc(x, *args):#считаем градиент функции costFunc
    #градиент помогает быстрей найти минимум?
    X, y, lam = args
    theta1 = np.reshape(x[0:HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)],
                        (HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE + 1), order='F')
    theta2 = np.reshape(x[HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1):],
                        (OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE + 1), order='F')
    m = np.size(X, 0)
    X = np.c_[np.ones((m, 1)), X]

    yk = np.zeros((OUTPUT_LAYER_SIZE, m))
    for i in range(m):
        yk[y[i] - 1, i] = 1

    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    for i in range(m):
        a1 = X[i, :]

        z2 = a1.dot(np.transpose(theta1))
        a2 = np.hstack((1, sigmoid(z2)))

        z3 = a2.dot(np.transpose(theta2))
        a3 = sigmoid(z3)

        delta_3 = a3 - np.transpose(yk[:, i])
        delta_2 = delta_3.dot(theta2) * sigmoid_gradient(np.hstack((1, z2)))
        delta_2 = delta_2[1:]

        theta2_grad = theta2_grad + np.outer(delta_3, a2)
        theta1_grad = theta1_grad + np.outer(delta_2, a1)

    theta1_grad[:, 0] = theta1_grad[:, 0] / m
    theta1_grad[:, 1:] = theta1_grad[:, 1:] / m + lam / m * theta1_grad[:, 1:]
    theta2_grad[:, 0] = theta2_grad[:, 0] / m
    theta2_grad[:, 1:] = theta2_grad[:, 1:] / m + lam / m * theta2_grad[:, 1:]

    return np.concatenate((theta1_grad.flatten('F'), theta2_grad.flatten('F')))

def predict(final_t'''готовые весы''', X):
    theta1 = np.reshape(final_t[0:HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)],
                        (HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE + 1), order='F')
    theta2 = np.reshape(final_t[HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1):],
                    (OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE + 1), order='F')
    m = np.size(X, 0)
    p = np.zeros((m, 1))

    h1 = sigmoid(np.c_[np.ones((m, 1)), X].dot(np.transpose(theta1)))
    h2 = sigmoid(np.c_[np.ones((m, 1)), h1].dot(np.transpose(theta2)))

    p = np.amax(h2, 1)#зачем это считать, если потом не используется?
    dummy = np.argmax(h2, 1)
    return dummy

data = np.load("data.npz")
X = data['arr_0']#множество картинок представленных в виде HOG
y = data['arr_1']#правильные ответы

mask = np.random.choice([False, True], len(X), p=[0.1, 0.9])#создаем маску, чтобы 
#разбить массив картинок на тренировку и тестировка 1(тест) к 9(тренировка)
training_X = X[mask, :]#картинки для тренировки
training_y = y[mask]#логотипы картинок для тренировки(для проверки)
testing_X = X[np.logical_not(mask), :]#картинки для тестинга
testing_y = y[np.logical_not(mask)]

init_t1 = initialize_theta(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE).flatten('F')#инициализируем весы на ребрах входного и внутреннего слоя
init_t2 = initialize_theta(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE).flatten('F')#инициализируем весы на ребрах внутреннего-выходного слоя
init_t = np.concatenate([init_t1, init_t2])#склейка двух массивов с весами?

itrCount = 0

final_t = optimize.fmin_cg(costFunc, init_t, fprime=gradFunc, args=(training_X, training_y, 0.05), full_output=1)#поиск минимальной ошибки?

pred = predict(final_t[0], testing_X) + 1#тестирование

np.save('thetas1', final_t[0])
#io.savemat('thetas.mat', {'t_1d': final_t[0]})
#print(np.mean(pred == testing_y))
