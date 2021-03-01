import math
import matplotlib.pyplot as plt
import numpy as np
import utility


def initial_state():
    return []


def predict(state, kernel, x_i):
    theta_x = 0.0
    state_length = len(state)
    if state_length != 0:
        for coefficient, x in state:
            theta_x += coefficient * kernel(x, x_i)
    return sign(theta_x)


def update_state(state, kernel, learning_rate, x_i, y_i):
    h_theta_x = predict(state, kernel, x_i)
    coefficient = learning_rate * (y_i - h_theta_x)
    state.append((coefficient, x_i))


def sign(a):
    if a >= 0:
        return 1
    return 0


def dot_kernel(a, b):
    return np.dot(a, b)


def rbf_kernel(a, b, sigma=1):
    return math.exp(-(a - b).dot(a - b) / (2 * (sigma ** 2)))


def train_perceptron(kernel_name, kernel, learning_rate):
    y_train, x_train = utility.load_csv("./input/ds5_train.csv")
    state = initial_state()
    for x_i, y_i in zip(x_train, y_train):
        update_state(state, kernel, learning_rate, x_i, y_i)

    # print(state)
    y_test, x_test = utility.load_csv("./input/ds5_test.csv")
    test_prediction = np.zeros(len(y_test))
    correct_predictions = 0

    for i, x in enumerate(x_test):
        test_prediction[i] = predict(state, kernel, x)
        if test_prediction[i] == y_test[i]:
            correct_predictions += 1
    print(test_prediction)
    print("Kernel " + kernel_name + " predicted correctly " + str(correct_predictions) + " out of " + str(len(y_test)) + " test samples")
    print("Success rate: " + str(correct_predictions / len(y_test)))


def main():
    train_perceptron('dot', dot_kernel, 0.25)
    train_perceptron('rbf', rbf_kernel, 0.25)


if __name__ == "__main__":
    main()

