# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split


def preprocessing(dataDir):

    data_list = []
    label_list = []
    dirs = os.listdir(dataDir)
    for dir in dirs:
        # 提取RFID数据
        # 三根天线，分别存储在num/first.csv second.csv third.csv中
        data_path = os.path.join(dataDir, dir, "data.csv")
        # print("reading: ", data_path)
        raw_df = pd.read_csv(data_path)
        selected_columns = raw_df[
            [
                "Ant1X",
                "Ant1Y",
                "Ant1Z",
                "Ant1Phase",
                "Ant2X",
                "Ant2Y",
                "Ant2Z",
                "Ant2Phase",
            ]
        ]

        # 提取数据存入训练集/测试集变量
        data = torch.tensor(selected_columns.values)
        data_list.append(data)

        # 提取标签数据
        label_path = os.path.join(dataDir, dir, "label.csv")
        df_label = pd.read_csv(label_path)
        label = df_label.values
        label_list.append(label)

    return data_list, label_list


if __name__ == "__main__":
    data, label = preprocessing(os.path.join("Transformer", "Dataset"))
    print(len(data), " ", len(label))
