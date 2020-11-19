import utility
import numpy as np


def main():
    directory = './data/ds2_train.csv'
    x_1, x_2, y = utility.load_csv(directory)
    x = [x_1, x_2]
    x = np.array(x)
    y = np.array(y)

    # -------- Data set is ready in array --------

    m = len(y)
    mu_0 = np.array([0, 0])
    mu_1 = np.array([0, 0])
    phi = y.sum() / m

    for i in range(m):
        if y[i]:
            mu_1 = mu_1 + x[:, i]
        else:
            mu_0 = mu_0 + x[:, i]
    mu_1 = mu_1 / y.sum()
    mu_0 = mu_0 / (m - y.sum())
    sigma_prime = x.copy()
    for i in range(m):
        if y[i] == 0:
            sigma_prime[0, i] = x[0, i] - mu_0[0]
            sigma_prime[1, i] = x[1, i] - mu_0[1]
        else:
            sigma_prime[0, i] = x[0, i] - mu_1[0]
            sigma_prime[1, i] = x[1, i] - mu_1[1]
    sigma = 1 / m * sigma_prime.dot(sigma_prime.T)

    # ------ Ready for prediction --------

    test_directory = './data/ds2_valid.csv'
    x_1, x_2, y = utility.load_csv(test_directory)
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    x = [x_1, x_2]
    x = np.array(x)

    prediction = []
    for i in range(len(y)):
        inv_sigma = inverse_calculator(sigma)
        # simplified probability
        p_x_y0 = np.exp(-0.5 * (x[:, i] - mu_0).T.dot(inv_sigma.dot(x[:, i] - mu_0)))
        p_x_y1 = np.exp(-0.5 * (x[:, i] - mu_1).T.dot(inv_sigma.dot(x[:, i] - mu_1)))
        if p_x_y1 * phi > p_x_y0 * (1-phi):
            prediction.append(1)
        else:
            prediction.append(0)

    f = open("results2.txt", "w")
    correct_prediction = 0
    for i in range(len(prediction)):
        f.write(str(prediction[i]))
        f.write(" ----- ")
        f.write(str(y[i]) + "\n")
        if y[i] == prediction[i]:
            correct_prediction += 1
    f.close()
    print(correct_prediction)

    print(mu_0)
    print(mu_1)


def inverse_calculator(matrix):
    return np.linalg.inv(matrix)


if __name__ == '__main__':
    main()
