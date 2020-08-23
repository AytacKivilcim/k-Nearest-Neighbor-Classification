import math
import numpy as np

iris_dataset = open("iris_dataset.txt", "r")
training_set = []
test_set = []
flower_counter = 0
for line in iris_dataset:
    flower_info = []
    if flower_counter == 50:
        flower_counter = 0
    splitedLine = line.split(",")
    sepal_length = float(splitedLine[0])
    petal_width = float(splitedLine[3])
    flower_type = splitedLine[4]
    if flower_type == "Iris-setosa\n":
        flower_type = 0
    elif flower_type == "Iris-versicolor\n":
        flower_type = 1
    else:
        flower_type = 2
    flower_info.append(sepal_length)
    flower_info.append(petal_width)
    flower_info.append(int(flower_type))
    if flower_counter < 30:
        training_set.append(flower_info)
    else:
        test_set.append(flower_info)
    flower_counter += 1
iris_dataset.close()
# print(training_set)
# print(test_set)

def distanceMetric(training_data_input, test_data_input, distance_type):
    if distance_type == "euclidean":
        result = math.sqrt(pow((training_data_input[0] - test_data_input[0]), 2) + pow((training_data_input[1] - test_data_input[1]), 2))
    elif distance_type == "manhattan":
        result = abs(training_data_input[0] - test_data_input[0]) + abs(training_data_input[1] - test_data_input[1])
    return result

def knnFuntion(k_value, training_set, test_set, distanceMetricType):
    predicted_labels = []
    for test_data in test_set:
        result_table = []
        guess_counts = [0, 0, 0]
        distances = []
        labels = []
        final_table = []
        for training_data in training_set:
            distance = distanceMetric(training_data, test_data, distanceMetricType)
            predicted_result = (distance, training_data[2])
            result_table.append(predicted_result)
        #print(result_table)
        for i in range(len(result_table)):
            distances.append(result_table[i][0])
            labels.append(result_table[i][1])
        for i in range(len(result_table)):
            current_min_index = np.argmin(distances)
            final_table.append((distances[current_min_index], labels[current_min_index]))
            del distances[current_min_index]
            del labels[current_min_index]
        # print(final_table)
        # final_table = sorted(result_table, key=lambda x: (x[0]))
        for i in range(k_value):
            if final_table[i][1] == 0:
                guess_counts[0] += 1
            elif final_table[i][1] == 1:
                guess_counts[1] += 1
            elif final_table[i][1] == 2:
                guess_counts[2] += 1
        #print(np.argmax(guess_counts))
        predicted_labels.append(np.argmax(guess_counts))
    return predicted_labels

def calculatePercentage(calculation_count, training_set, test_set):
    k = 1
    for j in range(calculation_count):
        test_set_labels = []
        for label in test_set:
            test_set_labels.append(label[2])
        labels = knnFuntion(k, training_set, test_set, "euclidean")
        #print(labels)
        #print(test_set_labels)
        hit_count = 0
        total_labels = 0
        for i in range(len(labels)):
            total_labels += 1
            if labels[i] == test_set_labels[i]:
                hit_count += 1
        print("k = {} for euclidean error count = {} percentage = {}".format(k, total_labels - hit_count, hit_count * 100 / total_labels))

        test_set_labels = []
        for label in test_set:
            test_set_labels.append(label[2])
        labels = knnFuntion(k, training_set, test_set, "manhattan")
        # print(labels)
        # print(test_set_labels)
        hit_count = 0
        total_labels = 0
        for i in range(len(labels)):
            total_labels += 1
            if labels[i] == test_set_labels[i]:
                hit_count += 1
        print("k = {} for manhattan error count = {} percentage = {}\n".format(k, total_labels-hit_count, hit_count * 100 / total_labels))
        k += 2

calculatePercentage(10, training_set, test_set)