import utility
import numpy as np


def get_words(message):
    lower_cased = message.lower()
    words_list = lower_cased.split()
    return words_list


def create_dictionary(messages):
    dictionary = {}
    for message in messages:
        words_list = get_words(message)
        for word in words_list:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1
    delete = [key for key in dictionary if dictionary[key] < 6]
    for key in delete: del dictionary[key]
    numeric_dict = {}
    for i, key in enumerate(dictionary):
        numeric_dict[key] = i
    return numeric_dict


def transform_text(messages, numeric_dictionary):
    messages_len = len(messages)
    dict_len = len(numeric_dictionary)
    array = np.zeros((messages_len, dict_len), dtype=int)

    for i, message in enumerate(messages):
        word_list = get_words(message)
        for word in word_list:
            if word in numeric_dictionary:
                array[i, numeric_dictionary[word]] += 1
    return array


def fit_naive_bayes_model(matrix, labels):
    # calculating th phi values
    y_0 = 0
    y_1 = 0
    dict_length = len(matrix[0])
    phi_j_y0 = np.zeros(dict_length)
    phi_j_y1 = np.zeros(dict_length)
    for i in labels:
        if i == "spam":
            y_1 += 1
    y_0 = len(labels) - y_1
    phi_y = y_1 / (y_1 + y_0)
    for i in range(len(labels)):
        for j in range(dict_length):
            if labels[i] == "spam":
                phi_j_y1[j] += matrix[i, j]
            else:
                phi_j_y0[j] += matrix[i, j]

    # Laplace Smoothing
    for j in range(dict_length):
        phi_j_y1[j] += 1
        phi_j_y0[j] += 1
    for j in range(dict_length):
        phi_j_y0[j] /= (y_0 + dict_length)
        phi_j_y1[j] /= (y_1 + dict_length)

    print(phi_y)
    print(phi_j_y0)
    print(phi_j_y1)
    return phi_y, phi_j_y0, phi_j_y1


def predict_from_naive_bayes_model(phi_y, phi_j_y0, phi_j_y1, matrix):
    data_length, dict_length = matrix.shape
    predictions = np.ones(data_length, dtype=bool)
    for i in range(data_length):
        prob_y_1 = np.log(phi_y)
        prob_y_0 = np.log(1 - phi_y)
        for j in range(dict_length):
            prob_y_1 += np.log(phi_j_y1[j]) * matrix[i, j]
            prob_y_0 += np.log(phi_j_y0[j]) * matrix[i, j]
        if prob_y_1 > prob_y_0:
            predictions[i] = True
        else:
            predictions[i] = False
    return predictions


def get_top_five_naive_bayes_words(model, dictionary):
    dict_length = len(dictionary)
    max_five = np.zeros(5)
    index = np.zeros(5)
    for i in range(dict_length):
        least = np.argmin(max_five)
        if model[1][i] / model[0][i] > max_five[least]:
            max_five[least] = model[1][i] / model[0][i]
            index[least] = i
    word_list = []
    for word, number in dictionary.items():
        for i in index:
            if number == i:
                word_list.append(word)
    return word_list


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    return


def main():
    train_label, train_messages = utility.load_tsv("./data/ds6_train.tsv")
    valid_label, valid_messages = utility.load_tsv("./data/ds6_val.tsv")
    test_label, test_messages = utility.load_tsv("./data/ds6_test.tsv")
    dictionary = create_dictionary(train_messages)
    utility.write_json('./dictionary', dictionary)
    train_matrix = transform_text(train_messages, dictionary)
    np.savetxt('./sample_train_matrix', train_matrix[:100, :])
    test_matrix = transform_text(test_messages, dictionary)
    np.savetxt('./sample_test_matrix', test_matrix[:100, :])
    val_matrix = transform_text(valid_messages, dictionary)

    phi_y, phi_j_y0, phi_j_y1 = fit_naive_bayes_model(train_matrix, train_label)

    naive_bayes_prediction_test = predict_from_naive_bayes_model(phi_y, phi_j_y0, phi_j_y1, test_matrix)
    correct_prediction = 0
    for i in range(len(test_label)):
        if test_label[i] == 'spam' and naive_bayes_prediction_test[i] == 1:
            correct_prediction += 1
        if test_label[i] == 'ham' and naive_bayes_prediction_test[i] == 0:
            correct_prediction += 1
    print("data size is " + str(len(test_label)) + " and correct results are " + str(correct_prediction))
    print("Correct prediction rate is " + str(correct_prediction / len(test_label)))
    np.savetxt('./naive_bayes_prediction_test', naive_bayes_prediction_test)
    top_five = get_top_five_naive_bayes_words([phi_j_y0, phi_j_y1], dictionary)
    print("The top 5 indicative words are: " + str(top_five))


if __name__ == '__main__':
    main()
