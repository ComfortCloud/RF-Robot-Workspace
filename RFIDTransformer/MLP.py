import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import torchvision
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import time
from sklearn import metrics
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import math
from sklearn import datasets  # 导入库
from sklearn.model_selection import train_test_split
from PreTraining import *
from torch.utils.data import Dataset


# 搭建全连接神经网络
class MLPPregression(nn.Module):  
    def __init__(self, layer_dims, dropout = 0.2):  
        super(MLPPregression, self).__init__()
        self.dropout = dropout
        self.feature_size = layer_dims[0]
        self.num_layers = len(layer_dims) - 1 
        # 定义所有隐藏层  
        self.layers = nn.ModuleList()  
        for i in range(len(layer_dims) - 1):  
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
            self.dropout = nn.Dropout(dropout)
        # 回归预测层，通常不需要激活函数，除非有特定需求  
        self.predict = nn.Linear(layer_dims[-2], layer_dims[-1], bias=True)
        
    def forward(self, x):  
        for layer in self.layers[:-1]:  # 对除了最后一层之外的所有层应用ReLU激活  
            x = F.relu(layer(x))  
        # 对最后一层（预测层）进行线性变换，通常不应用激活函数  
        output = self.predict(x)  
        return output
    
    def feature(self):
        return {
            "feature_size": self.feature_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

def reg_calculate(true, prediction, features=None):
    """
    To calculate the result of regression,
    including mse, rmse, mae, r2, four criterions.
    """
    if isinstance(true, torch.Tensor):
        true = true.numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.numpy()
    prediction[prediction < 0] = 0
    mse = metrics.mean_squared_error(true, prediction)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(true, prediction)
    mape = np.mean(np.abs((true - prediction) / true)) * 100
    r2 = metrics.r2_score(true, prediction)
    try:
        n = len(true)
        p = features
        r2_adjusted = 1 - ((1 - metrics.r2_score(true, prediction)) * (n - 1)) / (
            n - p - 1
        )
    except:
        print(
            "if you wanna get the value of r2_adjusted, you can define the number of features, "
            "which is the third parameter."
        )
        return mse, rmse, mae, mape, r2
    return mse, rmse, mae, mape, r2, r2_adjusted


# 创建数据生成器
def create_batch_size(X_train, y_train, batch_size):
    p = np.random.permutation(X_train.shape[0])
    data = X_train[p]
    label = y_train[p]

    batch_len = X_train.shape[0] // batch_size

    print([batch_size, batch_len])

    b_datas = []
    b_labels = []
    for i in range(batch_len):
        try:
            batch_data = data[batch_size * i : batch_size * (i + 1)]
            batch_label = label[batch_size * i : batch_size * (i + 1)]
        except:
            batch_data = data[batch_size * i : -1]
            batch_label = label[batch_size * i : -1]
        b_datas.append(batch_data)
        b_labels.append(batch_label)

    return b_datas, b_labels

'''
定义RFID数据集
'''
class RFIDDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        return image, label


if __name__ == "__main__":
    
    # 导入数据集
    data, label = preprocessing(os.path.join("/home/yangys/GeneratedData"))
    # 补全长度：padding
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    data = np.transpose(padded_sequences.numpy(), (0, 1, 2))
    print('The whole dataset size: {}'.format(data.shape))
    
    '''
    参数定义
    '''
    # 定义超参数
    output_dim = 3  # 输出向量维度（x,y,z
    max_seq_len = data.shape[1]
    input_dim = data.shape[2]
    
    # 初始化训练参数
    learning_rate = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    weight_decay = 5e-5
    dropout = 0.1
    batch_size = 128
    use_more_gpu = True
    save_path = "MLP_Result"
    epoch = 20000
    avgLossList = []  # put the avgLoss data
    TrainLosses = []
    TestLosses = []
    t = 0
    D = []
    n = 0  # 来记录 梯度衰减 的次数
    limit = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    device = [8]

    '''
    数据集准备
    '''
    # 切割数据样本集合测试集
    train_data, test_data, train_label, test_label = train_test_split(
        data, label, test_size=0.2
    )  # 20%测试集；80%训练集
    print('Trainset Size: {}, Testset Size: {}.'.format(len(train_data), len(test_data)))
    
    # 准备训练数据
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    # 调整label数据的格式：将行向量转换为列向量
    if train_label.ndim == 1:
        train_label = train_label.reshape(-1, 1)
    if test_label.ndim == 1:
        test_label = test_label.reshape(-1, 1)
    train_label = train_label.squeeze(1)
    test_label = test_label.squeeze(1)
    # 将矩阵展平为一维向量  
    train_data = train_data.reshape(len(train_data), -1)
    test_data = test_data.reshape(len(test_data), -1)
    print("Data shape:{}, Label shape:{}".format(train_data.shape, train_label.shape))
    
    # DataLoader
    # 准备训练数据集
    train_dataset = RFIDDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # 准备的测试数据集
    test_dataset = RFIDDataset(test_data, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    """
    训练模型
    """
    # 初始化模型
    layer_dims = [max_seq_len*input_dim, 1024, 1024, 512, 3]
    model = MLPPregression(layer_dims,dropout)  
    print(model)
    
    # 定义训练结果保存方式与路径
    resultDict = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epoch": epoch,
        "weight_decay": weight_decay,
        "use_more_gpu": use_more_gpu,
        "device": device,
    }
    resultDict = dict(resultDict, **model.feature())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_result = os.path.join(save_path, "Results.csv")
    try:
        count = len(open(save_result, "r").readlines())
    except:
        count = 1
    net_weight = os.path.join(save_path, "Weight")
    if not os.path.exists(net_weight):
        os.makedirs(net_weight)
    net_path = os.path.join(net_weight, str(count) + ".pkl")
    net_para_path = os.path.join(net_weight, str(count) + "_parameters.pkl")
    
    # 设置网络使用的硬件：cpu or gpu
    device_ids = [8] if torch.cuda.device_count() > 2 else [0]
    device = torch.device(device_ids[0] if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Let's use GPU: {}".format(device))
    else:
        print("Let's use CPU")
    model = model.to(device)
    model.train()

    # 设置优化器和损失函数
    try:
        optim = torch.optim.Adam(
            model.parameters(), lr=learning_rate[0], weight_decay=weight_decay
        )
    except:
        optim = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    criterion = torch.nn.MSELoss()

    # 正式开始训练
    start = time.time()  # 计算时间
    for e in range(epoch):
        tempLoss = []
        model.train()
        for batch_train_data, batch_train_label in train_loader:
            if torch.cuda.is_available():
                train_x = batch_train_data.to(device)
                train_y = batch_train_label.to(device)
            else:
                train_x = torch.FloatTensor(batch_train_data)
                train_y = torch.FloatTensor(batch_train_label)

            prediction = model(train_x)
            train_y = train_y.squeeze(1)
            loss = criterion(prediction, train_y)
            tempLoss.append(float(loss))

            optim.zero_grad()
            loss.backward()
            optim.step()

            D.append(loss.item() if loss.dim() == 0 else loss.cpu().numpy())
            avgloss = np.array(tempLoss).sum() / len(tempLoss)
            avgLossList.append(avgloss)

        if (e + 1) % 100 == 0:
            print("Training... epoch: {}, loss: {}".format((e + 1), avgLossList[-1]))

            # 此处进入model.eval模式，是为了计算阶段性的损失函数等数据，最后会作为模型训练记录写入文件
            # 无需使用 Variable，直接使用 Tensor  
            # 初始化累计损失  
            total_test_loss = 0.0
            with torch.no_grad():  # 使用 torch.no_grad() 来确保不计算梯度，节省内存和计算资源  
                model.eval()
                for batch_test_data, batch_test_label in test_loader:
                    if torch.cuda.is_available():  
                        test_x = batch_test_data.to(device)  
                        test_y = batch_test_label.to(device)  
                    else:  
                        test_x = batch_test_data
                        test_y = batch_test_label
                    test_prediction = model(test_x)  # 这里可能出现问题，确保输入与位置编码兼容
            
                    test_y = test_y.squeeze(1)
                    test_loss = criterion(test_prediction, test_y)
                    # 累计损失  
                    total_test_loss += test_loss.item()
            
            TrainLosses.append(avgloss)
            TestLosses.append(total_test_loss)

        # epoch 终止装置
        if len(D) >= 20:
            loss1 = np.mean(np.array(D[-20:-10]))
            loss2 = np.mean(np.array(D[-10:]))
            d = float(np.abs(loss2 - loss1))  # 计算loss的差值

            if (
                d < limit[n] or e == epoch - 1 or e > (epoch - 1) / 3 * (n + 1)
            ):  # 加入遍历完都没达成limit限定，就直接得到结果

                D = []  # 重置
                print("The error changes within {}".format(limit[n]))
                n += 1
                e = e + 1
                print(
                    "Training... epoch: {}, loss: {}".format(
                        (e + 1), loss.cpu().data.numpy()
                    )
                )

                torch.save(model, net_path)
                torch.save(model.state_dict(), net_para_path)

                # 初始化累计损失  
                total_test_loss = 0.0
                test_predictions = []
                with torch.no_grad():  # 使用 torch.no_grad() 来确保不计算梯度，节省内存和计算资源  
                    model.eval()
                    for batch_test_data, batch_test_label in test_loader:
                        if torch.cuda.is_available():  
                            test_x = batch_test_data.to(device)  
                            test_y = batch_test_label.to(device)  
                        else:  
                            test_x = batch_test_data
                            test_y = batch_test_label
                        # 这里可能出现问题，确保输入与位置编码兼容
                        # print(test_x.shape)
                        test_prediction = model(test_x)
                
                        test_y = test_y.squeeze(1)
                        test_loss = criterion(test_prediction, test_y)
                        # 累计损失
                        total_test_loss += test_loss.item()

                        test_prediction = test_prediction.cpu().data.numpy()
                        test_predictions.append(test_prediction)
                        test_y = test_y.cpu()
                
                test_predictions = np.concatenate(test_predictions, axis=0)
                # print(y_test.shape,test_predictions.shape)

                (
                    mse,
                    rmse,
                    mae,
                    mape,
                    r2,
                    r2_adjusted,
                    # self.rmsle,
                ) = reg_calculate(test_label, test_predictions, test_data.shape[-1])

                # test_acc = self.__get_acc(test_prediction, test_y)
                print(
                    "\033[1;35m Testing... epoch: {}, loss: {} , r2 {}\033[0m!".format(
                        (e + 1), test_loss.cpu().data.numpy(), r2
                    )
                )

                # 已经梯度衰减了n次
                if n == len(limit):
                    print("The meaning of the loop is not big, stop!!")
                    break
                print("Now learning rate is : {}".format(learning_rate[n]))
                optim.param_groups[0]["lr"] = learning_rate[n]

    end = time.time()
    t = end - start
    print("Training completed!!! Time consuming: {}".format(str(t)))

    #
    resDict = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "r2_adjusted": r2_adjusted,
    }
    resultDict = dict(resDict, **resultDict)

    # 计算结果
    (
        mse,
        rmse,
        mae,
        mape,
        r2,
        r2_adjusted,
        # self.rmsle,
    ) = reg_calculate(test_label, test_predictions, test_data.shape[-1])

    # 保存参数：预测值，真实值
    resultTitle = [str(line) for line in resultDict.keys()]
    resultList = [
        "_".join([str(l) for l in line]) if isinstance(line, list) else str(line)
        for line in resultDict.values()
    ]

    # 计算行数，匹配 prediciton 的保存
    save_result = "/".join([save_path, "result.csv"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        count = len(open(save_result, "r").readlines())
    except:
        count = 1

    # 判断是否存在未见 没有则写入文件 有则追加写入
    resultTitle.insert(0, "count")
    resultList.insert(0, str(count))

    if not os.path.exists(save_result):
        with open(save_result, "w") as f:
            titleStr = ",".join(resultTitle)
            f.write(titleStr)
            f.write("\n")

    with open(save_result, "a+") as f:
        contentStr = ",".join(resultList)
        f.write(contentStr)
        f.write("\n")

    # 保存 train loss 和 test loss
    Loss_path = os.path.join(save_path, "Loss")
    if not os.path.exists(Loss_path):
        os.makedirs(Loss_path)

    save_Loss = os.path.join(Loss_path, str(count) + ".csv")

    df = pd.DataFrame()
    df["TrainLoss"] = TrainLosses
    df["TestLoss"] = TestLosses
    df.to_csv(save_Loss, index=False)

    # 保存 prediction
    pred_path = os.path.join(save_path, "Prediction")
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    save_prediction = os.path.join(pred_path, str(count) + ".csv")
    df = pd.DataFrame()

    df["test_label"] = [i for i in test_label]
    df["test_prediction"] = [i for i in test_predictions]
    df.to_csv(save_prediction, index=False)

    print("Save the value of prediction successfully!!")

    # 保存模型权重
    model_path = os.path.join(save_path, "Model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if use_more_gpu:
        torch.save(model.state_dict(), os.path.join(model_path, str(count) + ".pth"))
    else:
        torch.save(
            model.net.state_dict(), os.path.join(model_path, str(count) + ".pth")
        )

    print("Save the model weight successfully!!")
