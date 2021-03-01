import numpy as np
import matplotlib.pyplot as plt


def load_csv(csv_address):
    my_data = np.genfromtxt(csv_address, delimiter=',')
    num_of_rows = len(my_data) - 1
    num_of_cols = len(my_data[1])
    # print("num of cols are: " + str(num_of_cols))
    # print("num of rows are : " + str(num_of_rows))
    x = np.zeros((num_of_rows, num_of_cols - 1))
    y = np.zeros(num_of_rows)
    for i in range(0, num_of_rows):
        for j in range(0, num_of_cols - 1):
            x[i, j] = my_data[i + 1][j + 1]
        y[i] = my_data[i + 1][0]
    return y, x


def show_plot(x_1, x_2, y, theta_0, theta_1, theta_2, save_path):

    # plotting points as a scatter plot
    plt.scatter(x_1[y == 1], x_2[y == 1], color="green", s=5)
    plt.scatter(x_1[y == 0], x_2[y == 0], color="blue", s=5)
    plt.xlabel('x_1 - axis')
    plt.ylabel('x_2 - axis')

    x_sample = [min(x_1)]
    y_sample = [-(theta_0 + theta_1 * x_sample[0])/theta_2]
    x_sample.append(max(x_1))
    y_sample.append(-(theta_0 + theta_1 * x_sample[1])/theta_2)
    plt.plot(x_sample, y_sample, color="red")
    plt.title('Training Samples!')
    #plt.show()
    plt.savefig('plot.png', bbox_inches='tight')
