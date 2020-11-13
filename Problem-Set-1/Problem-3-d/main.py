import utility as util
import numpy as np


def main():
    x = []
    x1, x2, x3, x4, y = util.load_csv('./data/ds4_train.csv')
    x0 = []
    for i in range(len(y)):
        x0.append(1)
    x.append(np.array(x0))
    x.append(x1)
    x.append(x2)
    x.append(x3)
    x.append(x4)
    theta = []
    for i in range(5):
        theta.append(0)

    # ---- theta[4] is multiplied to x4 -----
    alpha = 0.00000007
    while True:
        # --- Continue the loop until it converges ---
        before_theta = theta.copy()
        for k in range(5):
            sigma = 0
            for i in range(len(y)):
                sigma = sigma + (y[i] - h_theta_x(theta, x, i)) * x[k][i]
            theta[k] = theta[k] + alpha * sigma / len(y)
        print(theta)
        delta = 0
        for k in range(5):
            delta = delta + abs(theta[k] - before_theta[k])
        print(delta)
        if delta < 0.01:
            break

    # ----- End of learning ----

    x1, x2, x3, x4, y = util.load_csv('./data/ds4_valid.csv')
    x_valid = [np.array(x0), x1, x2, x3, x4]
    f = open("results.txt", "w")
    prediction = []
    total_diff = 0
    for i in range(len(y)):
        prediction.append(h_theta_x(theta, x_valid, i))
        f.write(str(prediction[i]))
        f.write("  -----  ")
        f.write(str(y[i]))
        f.write("  -----  ")
        f.write(str(abs(prediction[i] - y[i])) + "\n")
        total_diff = total_diff + prediction[i] - y[i]
    avg_prediction_diff = total_diff / len(y)
    avg_y = np.sum(y) / len(y)
    f.write("Average error is: " + str(avg_prediction_diff) + "\n")
    f.write("Average of y is: " + str(avg_y) + "\n")
    f.write("Average error percentage is: " + str(avg_prediction_diff / avg_y * 100))
    f.close()


def h_theta_x(theta, x, i):
    return np.exp(theta[0] * x[0][i] + theta[1] * x[1][i] + theta[2] * x[2][i] + theta[3] * x[3][i] + theta[4] * x[4][i])


if __name__ == '__main__':
    main()
