import utility
import numpy as np


def main():
    directory = './data/ds5_train.csv'
    x_train, y_train = utility.load_csv(directory)

    # -------- Data set is ready in array --------

    directory = './data/ds5_valid.csv'
    x_valid, y_valid = utility.load_csv(directory)

    valid_results = []
    for i in range(len(x_valid)):
        # --- Calculation of W matrix ---
        w_arr = []
        for j in range(len(x_train)):
            w_arr.append(0.5 * np.exp(-pow(x_train[j][1] - x_valid[i][1], 2) / (2 * 0.5 * 0.5)))
        w_arr = np.array(w_arr)
        W = np.diag(w_arr)
        theta = inverse_calculator(x_train.T.dot(W).dot(x_train)).dot(x_train.T).dot(W).dot(y_train)
        valid_results.append(x_valid[i].dot(theta))

    f = open("valid_results.txt", "w")
    sum_of_diff = 0
    for i in range(len(valid_results)):
        f.write(str(valid_results[i]))
        f.write(" ----- ")
        f.write(str(y_valid[i]) + "\n")
        sum_of_diff = sum_of_diff + pow(valid_results[i] - y_valid[i], 2)
    mse = sum_of_diff / len(y_valid)
    f.write("*** MSE value for valid data is: " + str(mse) + " ***\n")
    f.close()
    utility.show_plot(x_valid, valid_results, x_train, y_train)

def inverse_calculator(matrix):
    return np.linalg.inv(matrix)


if __name__ == '__main__':
    main()
