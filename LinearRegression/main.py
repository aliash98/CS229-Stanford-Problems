import numpy as np


def main():
    my_training_data = np.genfromtxt("./data/ds1_train.csv", delimiter=',')
    # print(type(my_data))
    # print(my_data.shape)
    x_train = my_training_data[1:, :2]
    # we need to add a column of 1's to x_train
    x_1 = np.ones(800)
    x_train = np.insert(x_train, 0, x_1, axis=1)
    y_train = my_training_data[1:, 2]
    print(x_train.shape)
    print(y_train.shape)
    # print(x_train)
    # print(y_train)

    theta = np.ones(3)

    # Starting to learn theta for

    x_t = x_train.transpose()

    # print(np.linalg.inv(np.matmul(x_train, x_T)).shape)
    theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_t, x_train)), x_t), y_train)
    print(theta.shape)
    print(theta)

    # lets test theta

    my_test_data = np.genfromtxt("./data/ds1_valid.csv", delimiter=',')
    x_test = my_test_data[1:, :2]
    x_1 = np.ones(100)
    x_test = np.insert(x_test, 0, x_1, axis=1)
    y_test = my_test_data[1:, 2]
    predictions = np.matmul(x_test, theta)
    print(predictions)
    print(predictions.shape)
    correct_predictions = 0
    for i in range(100):
        if (predictions[i] < 0.5 and y_test[i] == 0) or (predictions[i] >= 0.5 and y_test[i] == 1):
            correct_predictions += 1
    print("Linear Regression is predicting " + str(correct_predictions) + " percent of test samples correctly.")


if __name__ == '__main__':
    main()

