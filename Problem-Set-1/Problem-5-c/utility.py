import numpy as np
import matplotlib.pyplot as plt


def load_csv(csv_address):
    my_data = np.genfromtxt(csv_address, delimiter=',')
    num_of_rows = len(my_data) - 1
    num_of_cols = len(my_data[1])

    x = []
    y = []
    for i in range(1, num_of_rows+1):
        temp = [1, my_data[i][0]]
        x.append(temp)
        y.append(my_data[i][1])
    x = np.array(x)
    return x, np.array(y)


def show_plot(x_valid, valid_results, x_train, y_train, filename):

    # plotting points as a scatter plot
    plt.clf()
    plt.scatter(x_train[:, 1], y_train, color="green", s=5)
    plt.scatter(x_valid[:, 1], valid_results, color="blue", s=5)
    plt.xlabel('X - axis')
    plt.ylabel('Y - axis')
    plt.title('Training Samples!')
    #plt.show()
    plt.savefig(filename + '.png', bbox_inches='tight')
