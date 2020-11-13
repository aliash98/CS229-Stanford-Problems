import utility
import numpy as np


def main():
    epsilon = 0.00000001
    directory = './data/ds1_train.csv'
    x_1, x_2, y = utility.load_csv(directory)
    x_inputs = np.array([x_1, x_2])
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    y = np.array(y)

    # -------- Data set is ready in array --------

    theta_0 = 0.01
    theta_1 = 0.01
    theta_2 = 0.01
    while True:
        gradient = gradient_calculator(x_1, x_2, y, theta_0, theta_1, theta_2)
        hessian = hessian_calculator(x_1, x_2, theta_0, theta_1, theta_2)
        hessian_inv = inverse_calculator(hessian)
        result = np.matmul(hessian_inv, gradient)
        if np.sum(result) < epsilon:
            break
        theta_2 += result[0]
        theta_1 += result[1]
        theta_0 += result[2]

    print(theta_2, " ", theta_1, " ", theta_0, "\n")
    utility.show_plot(x_1, x_2, y, theta_0, theta_1, theta_2, './plot.png')
    # ------ Training is finished -----------

    test_directory = './data/ds1_valid.csv'
    x_1, x_2, y = utility.load_csv(test_directory)
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    correct_prediction = 0
    valid_results = sigmoid_function(theta_0, theta_1, theta_2, x_1, x_2)
    f = open("results.txt", "w")
    for i in range(len(valid_results)):
        f.write(str(valid_results[i]))
        f.write(" ----- ")
        if valid_results[i] > 0.5:
            f.write("1\n")
            if y[i] == 1:
                correct_prediction += 1
        else:
            f.write("0\n")
            if y[i] == 0:
                correct_prediction += 1
    f.close()
    print(correct_prediction)


def sigmoid_function(theta_0, theta_1, theta_2, x_1, x_2):
    return 1/(1 + np.exp(theta_0 + theta_1 * x_1 + theta_2 * x_2))


def gradient_calculator(x_1, x_2, y, theta_0, theta_1, theta_2):
    sigmoid_result = np.array(sigmoid_function(theta_0, theta_1, theta_2, x_1, x_2))
    g_1 = np.sum((y - sigmoid_result) * x_2)
    g_2 = np.sum((y - sigmoid_result) * x_1)
    g_3 = np.sum(y - sigmoid_result)
    gradient = np.array([g_1, g_2, g_3])
    return gradient


def hessian_calculator(x_1, x_2, theta_0, theta_1, theta_2):
    sigmoid_result = np.array(sigmoid_function(theta_0, theta_1, theta_2, x_1, x_2))
    h_11 = - np.sum(sigmoid_result * (1 - sigmoid_result) * x_2 * x_2)
    h_12 = - np.sum(sigmoid_result * (1 - sigmoid_result) * x_2 * x_1)
    h_22 = - np.sum(sigmoid_result * (1 - sigmoid_result) * x_1 * x_1)
    h_13 = - np.sum(sigmoid_result * (1 - sigmoid_result) * x_2)
    h_23 = - np.sum(sigmoid_result * (1 - sigmoid_result) * x_1)
    h_33 = - np.sum(sigmoid_result * (1 - sigmoid_result))
    hessian = np.array([[h_11, h_12, h_13], [h_12, h_22, h_23], [h_13, h_23, h_33]])
    return hessian


def inverse_calculator(matrix):
    return np.linalg.inv(matrix)


if __name__ == '__main__':
    main()
