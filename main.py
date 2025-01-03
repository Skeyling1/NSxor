import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


I1 = 1
I2 = 0

w1, w2, w3, w4, w5, w6 = 0.45, 0.78, -0.12, 0.13, 1.5, -2.3

H1_input = I1 * w1 + I2 * w3
H2_input = I1 * w2 + I2 * w4

H1_output = sigmoid(H1_input)
H2_output = sigmoid(H2_input)

O1_input = H1_output * w5 + H2_output * w6

O1_output = sigmoid(O1_input)
error = (1 - O1_output) ** 2 / 1

print("result: ", O1_output)
print("error: ", error)