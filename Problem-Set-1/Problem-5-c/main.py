import utility
import numpy as np


def main():
    directory = './data/ds5_train.csv'
    x_train, y_train = utility.load_csv(directory)

    # -------- Data set is ready in array --------

    directory = './data/ds5_valid.csv'
    x_valid, y_valid = utility.load_csv(directory)

    tau = [0.03, 0.05, 0.1, 0.5, 1, 10]

    for k in range(6):
        valid_results = []
        for i in range(len(x_valid)):
            # --- Calculation of W matrix ---
            w_arr = []
            for j in range(len(x_train)):
                w_arr.append(0.5 * np.exp(-pow(x_train[j][1] - x_valid[i][1], 2) / (2 * tau[k] * tau[k])))
            w_arr = np.array(w_arr)
            W = np.diag(w_arr)
            theta = inverse_calculator(x_train.T.dot(W).dot(x_train)).dot(x_train.T).dot(W).dot(y_train)
            valid_results.append(x_valid[i].dot(theta))

        sum_of_diff = 0
        for i in range(len(valid_results)):
            sum_of_diff = sum_of_diff + pow(valid_results[i] - y_valid[i], 2)
        mse = sum_of_diff / len(y_valid)
        print("MSE for tau = " + str(tau[k]) + " is : " + str(mse), end="\n")
        utility.show_plot(x_valid, valid_results, x_train, y_train, "tau_is_" + str(tau[k]))


def inverse_calculator(matrix):
    return np.linalg.inv(matrix)


if __name__ == '__main__':
    main()
