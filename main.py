# supervised learning, backpropagation

import math
import random


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

E = 0.7 #learn_rate
A = 0.3 #momentum
delta_w1 = 0
delta_w2 = 0
delta_w3 = 0
delta_w4 = 0
delta_w5 = 0
delta_w6 = 0


training_sets = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]] #to find results of XOR - exclusive or


I1 = 1
I2 = 0

w1, w2, w3, w4, w5, w6 = [random.random() for w in range(6)]  #случайные веса
error = 1
number_of_itterations = 0

while error > 0.01:
    H1_input = I1 * w1 + I2 * w3
    H2_input = I1 * w2 + I2 * w4

    H1_output = sigmoid(H1_input)
    H2_output = sigmoid(H2_input)

    O1_input = H1_output * w5 + H2_output * w6

    O1_output = sigmoid(O1_input) #результат работы нейросети
    error = (1 - O1_output) ** 2 / 1   #Mean Squared Error отличие от 1 -правильного ответа

    print("w1: ", w1)
    print("w2: ", w2)
    print("w3: ", w3)
    print("w4: ", w4)
    print("w5: ", w5)
    print("w6: ", w6)
    print("result: ", O1_output)
    print("error: ", error)

    # погрешности нейронов, передаем погрешность от последнего
    delta_O1 = (1 - O1_output) * ((1 - O1_output) * O1_output)

    delta_H1 = ((1 - H1_output) * H1_output) * (w5 * delta_O1)
    delta_H2 = ((1 - H2_output) * H2_output) * (w6 * delta_O1)

    GRAD_w5 = H1_output * delta_O1
    GRAD_w6 = H2_output * delta_O1

    # поправки весов
    delta_w5 = E * GRAD_w5 + A * delta_w5
    delta_w6 = E * GRAD_w6 + A * delta_w6
    w5 = w5 + delta_w5
    w6 = w6 + delta_w6

    # не нужно находить дельты для входных нейронов так как у них нет входных синапсов.
    GRAD_w1 = I1 * delta_H1
    GRAD_w2 = I1 * delta_H2
    GRAD_w3 = I2 * delta_H1
    GRAD_w4 = I2 * delta_H2

    # поправки весов входных нейронов
    delta_w1 = E * GRAD_w1 + A * delta_w1
    delta_w2 = E * GRAD_w2 + A * delta_w2
    delta_w3 = E * GRAD_w3 + A * delta_w3
    delta_w4 = E * GRAD_w4 + A * delta_w4
    w1 = w1 + delta_w1
    w2 = w2 + delta_w2
    w3 = w3 + delta_w3
    w4 = w4 + delta_w4




    number_of_itterations += 1

    print("itteration: ", number_of_itterations)
















