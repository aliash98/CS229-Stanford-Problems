import csv
import json

import numpy as np
import matplotlib.pyplot as plt


def load_tsv(tsv_address):

    messages = []
    y = []
    tsv_file = open(tsv_address)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        if len(row) > 0:
            y.append(row[0])
            messages.append(row[1])
    tsv_file.close()
    return y, messages


def write_json(filename, value):
    with open(filename, 'w') as f:
        json.dump(value, f)


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
