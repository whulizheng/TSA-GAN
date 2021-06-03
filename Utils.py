import numpy as np
import pandas as pd
import json


def readjson(address):
    with open(address, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def readucr(filename):
    data = pd.read_csv(filename, header=None)
    Y = np.array(data[0])
    X = np.array(data)[:, 1:]
    return X, Y


def read_dataset(file_dir, dataset_name):
    datasets_dict = {}
    x_train, y_train = readucr(file_dir)
    datasets_dict[dataset_name] = (
        x_train.copy(), y_train.copy())
    return datasets_dict


def adjust_augmentation_auto(tags):
    tag_list = set(tags)
    average_num = len(tags)/len(tag_list)
    if average_num <= 64:
        average_num = 64
    augmentation_percentage_list = []
    for tag in tag_list:
        if tags.count(tag) >= average_num*0.95:
            augmentation_percentage_list.append(0)
        else:
            augmentation_percentage_list.append(
                round((average_num-tags.count(tag))/tags.count(tag), 1))
    return augmentation_percentage_list


def savetofile(dataset_dict, dataset_name, output_dir):
    x_train = dataset_dict[dataset_name][0]
    y_train = dataset_dict[dataset_name][1]
    additional_name = ''
    dataframe = pd.DataFrame()
    dataframe[0] = y_train.astype(np.int16)
    for i in range(len(x_train[0])):
        dataframe[i + 1] = x_train[:, i]
    dataframe.to_csv(output_dir + additional_name, index=None, header=None)
