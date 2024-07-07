import os
import torch
import torch.nn as nn
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


#  模型搭建：位置编码 Position Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):

        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransAm(nn.Module):
    def __init__(
        self,
        input_dim=9,
        feature_size=512,
        output_dim=3,
        num_layers=6,
        nhead=8,
        dropout=0.1,
        max_len=200,
    ):
        super(TransAm, self).__init__()
        self.model_type = "Transformer"

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_dim, max_len)

        print([input_dim, nhead])
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Linear(input_dim * max_len, output_dim)
        self.init_weights()

        self.feature_size = feature_size
        self.num_layers = num_layers
        self.dropout = dropout

    def feature(self):
        return {
            "feature_size": self.feature_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            # mask = self._generate_square_subsequent_mask(len(src)).to(device)
            # self.src_mask = mask
        src_key_padding_mask = src == 0
        src_key_padding_mask = src_key_padding_mask[:, :, 0]
        # print([src.shape, src_key_padding_mask.shape])
        src = self.pos_encoder(src)

        # 这里因为使用Transformer做回归，decoder部分是一个全连接层，而非生成式，因此不需要mask
        # output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        # print([src.shape, src_key_padding_mask.shape])
        # output = self.transformer_encoder(
        #     src=src, mask=None, src_key_padding_mask=src_key_padding_mask
        # )
        output = self.transformer_encoder(
            src=src, mask=None, src_key_padding_mask=src_key_padding_mask
        )
        output = output.view(output.shape[0], -1)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class General_Regression_Training_3d:
    # 初始化函数
    def __init__(
        self,
        net,
        learning_rate=[1e-3, 1e-5, 1e-7],
        batch_size=32,
        epoch=2000,
        use_more_gpu=False,
        weight_decay=1e-8,
        device=0,
        save_path="Transformer_Result",
    ):

        self.net = net
        self.resultDict = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epoch": epoch,
            "weight_decay": weight_decay,
            "use_more_gpu": use_more_gpu,
            "device": device,
        }
        self.resultDict = dict(self.resultDict, **self.net.feature())

        self.batch_size = batch_size
        self.use_more_gpu = use_more_gpu
        self.lr = learning_rate
        self.epoch = epoch
        self.weight_decay = weight_decay
        self.device = device
        self.epoch = epoch

        self.save_path = save_path  # 设置一条保存路径，直接把所有的值都收藏起来
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.avgLossList = []  # put the avgLoss data
        self.TrainLosses = []
        self.TestLosses = []
        self.t = 0
        self.D = []
        self.n = 0  # 来记录 梯度衰减 的次数
        self.limit = [1e-5, 1e-6, 1e-7]

    # 训练模型
    def fit(self, X_train, y_train, X_test, y_test):
        """training the network"""
        # input the dataset and transform into dataLoad
        # if y is a scalar
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        self.X_train, self.X_test, self.y_train, self.y_test = (
            X_train,
            X_test,
            y_train,
            y_test,
        )

        b_data, b_labels = self.create_batch_size(X_train, y_train)

        # 定义模型训练结果输出路径
        save_result = os.path.join(self.save_path, "Results.csv")
        try:
            count = len(open(save_result, "rU").readlines())
        except:
            count = 1

        net_weight = os.path.join(self.save_path, "Weight")
        if not os.path.exists(net_weight):
            os.makedirs(net_weight)

        net_path = os.path.join(net_weight, str(count) + ".pkl")
        net_para_path = os.path.join(net_weight, str(count) + "_parameters.pkl")

        # 设置网络使用的硬件：cpu or gpu
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Let's use GPU: {}".format(self.device))
        else:
            print("Let's use CPU")

        if self.use_more_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs")
            # dim = 0 [64, xxx] -> [32, ...], [32, ...] on 2GPUs
            self.net = nn.DataParallel(self.net)
        self.net.to(device)

        # network change to train model
        self.net.train()
        # 设置优化器和损失函数
        try:
            optim = torch.optim.Adam(
                self.net.parameters(), lr=self.lr[0], weight_decay=self.weight_decay
            )
        except:
            optim = torch.optim.Adam(
                self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        criterion = torch.nn.MSELoss()
        print("")
        # Officially start training

        # 开始训练
        start = time.time()  # 计算时间
        limit = self.limit[0]
        for e in range(self.epoch):

            tempLoss = []
            # 训练模型
            self.net.train()
            for i in range(len(b_data)):
                if torch.cuda.is_available():
                    # print('cuda')
                    # self.net = self.net.cuda()
                    train_x = Variable(torch.FloatTensor(b_data[i])).to(device)
                    train_y = Variable(torch.FloatTensor(b_labels[i])).to(device)
                else:
                    train_x = Variable(torch.FloatTensor(b_data[i]))
                    train_y = Variable(torch.FloatTensor(b_labels[i]))

                # prediction = self.net(train_x, src_key_padding_mask)
                prediction = self.net(train_x)
                train_y = train_y.squeeze(1)
                # print("squeeze here 1.")
                loss = criterion(prediction, train_y)
                tempLoss.append(float(loss))

                optim.zero_grad()
                loss.backward()
                optim.step()

            self.D.append(loss.cpu().data.numpy())
            avgloss = np.array(tempLoss).sum() / len(tempLoss)
            self.avgLossList.append(avgloss)

            if (e + 1) % 100 == 0:
                print(
                    "Training... epoch: {}, loss: {}".format(
                        (e + 1), self.avgLossList[-1]
                    )
                )

                self.net.eval()
                if torch.cuda.is_available():
                    test_x = Variable(torch.FloatTensor(self.X_test)).to(device)
                    test_y = Variable(torch.FloatTensor(self.y_test)).to(device)
                else:
                    test_x = Variable(torch.FloatTensor(self.X_test))
                    test_y = Variable(torch.FloatTensor(self.y_test))

                test_prediction = self.net(test_x)
                test_y = test_y.squeeze(1)
                # print("squeeze here 2.")
                test_loss = criterion(test_prediction, test_y)

                self.TrainLosses.append(avgloss)
                self.TestLosses.append(test_loss.cpu().data.numpy())

                self.test_prediction = test_prediction.cpu().data.numpy()
                self.test_prediction[self.test_prediction < 0] = 0

            # epoch 终止装置
            if len(self.D) >= 20:
                loss1 = np.mean(np.array(self.D[-20:-10]))
                loss2 = np.mean(np.array(self.D[-10:]))
                d = float(np.abs(loss2 - loss1))  # 计算loss的差值

                if (
                    d < limit
                    or e == self.epoch - 1
                    or e > (self.epoch - 1) / 3 * (self.n + 1)
                ):  # 加入遍历完都没达成limit限定，就直接得到结果

                    # if e == self.epoch - 1 or e > (self.epoch - 1) / 3 * (
                    #     self.n + 1
                    # ):  # 加入遍历完都没达成limit限定，就直接得到结果
                    self.D = []  # 重置
                    self.n += 1
                    print("The error changes within {}".format(limit))
                    self.e = e + 1

                    # train_acc = self.__get_acc(prediction, train_y)
                    print(
                        "Training... epoch: {}, loss: {}".format(
                            (e + 1), loss.cpu().data.numpy()
                        )
                    )

                    # torch.save(self.net.module.state_dict(), model_out_path) 多 GPU 保存

                    torch.save(self.net, net_path)
                    torch.save(self.net.state_dict(), net_para_path)

                    self.net.eval()
                    if torch.cuda.is_available():
                        test_x = Variable(torch.FloatTensor(self.X_test)).to(device)
                        test_y = Variable(torch.FloatTensor(self.y_test)).to(device)
                    else:
                        test_x = Variable(torch.FloatTensor(self.X_test))
                        test_y = Variable(torch.FloatTensor(self.y_test))

                    test_prediction = self.net(test_x)
                    test_y = test_y.squeeze(1)

                    # print("squeeze here 3.")
                    test_loss = criterion(test_prediction, test_y)

                    self.test_prediction = test_prediction.cpu().data.numpy()
                    self.test_prediction[self.test_prediction < 0] = 0

                    test_y = test_y.cpu()
                    print("self.test_y", np.array(test_y).shape)
                    print("self.test_prediction", self.test_prediction.shape)
                    print("self.test_prediction", "self.X_test.shape[-1]")
                    print(self.test_prediction, test_y)

                    (
                        self.mse,
                        self.rmse,
                        self.mae,
                        self.mape,
                        self.r2,
                        self.r2_adjusted,
                        # self.rmsle,
                    ) = self.reg_calculate(
                        test_y, self.test_prediction, self.X_test.shape[-1]
                    )

                    # test_acc = self.__get_acc(test_prediction, test_y)
                    print(
                        "\033[1;35m Testing... epoch: {}, loss: {} , r2 {}\033[0m!".format(
                            (e + 1), test_loss.cpu().data.numpy(), self.r2
                        )
                    )

                    # 已经梯度衰减了 2 次
                    if self.n == 3:
                        print("The meaning of the loop is not big, stop!!")
                        break
                    limit = self.limit[self.n]
                    print("Now learning rate is : {}".format(self.lr[self.n]))
                    optim.param_groups[0]["lr"] = self.lr[self.n]

        end = time.time()
        self.t = end - start
        print("Training completed!!! Time consuming: {}".format(str(self.t)))

        #
        resDict = {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "r2_adjusted": self.r2_adjusted,
            # "rmsle": self.rmsle,
        }
        self.resultDict = dict(resDict, **self.resultDict)

        # 计算结果
        (
            self.mse,
            self.rmse,
            self.mae,
            self.mape,
            self.r2,
            self.r2_adjusted,
            # self.rmsle,
        ) = self.reg_calculate(test_y, self.test_prediction, self.X_test.shape[-1])

    # 保存参数  预测值 真实值
    def save_results(self):
        # , resultTitle, resultList, y_test, test_prediction, save_path
        resultTitle = [str(line) for line in self.resultDict.keys()]
        resultList = [
            "_".join([str(l) for l in line]) if isinstance(line, list) else str(line)
            for line in self.resultDict.values()
        ]
        y_test = self.y_test
        test_prediction = self.test_prediction
        save_path = self.save_path

        # 计算行数，匹配 prediciton 的保存
        save_result = "/".join([save_path, "result.csv"])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        try:
            count = len(open(save_result, "rU").readlines())
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
        df["TrainLoss"] = self.TrainLosses
        df["TestLoss"] = self.TestLosses
        df.to_csv(save_Loss, index=False)
        # 保存 prediction
        pred_path = os.path.join(save_path, "Prediction")
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        save_prediction = os.path.join(pred_path, str(count) + ".csv")
        df = pd.DataFrame()

        df["y_test"] = [i for i in y_test]
        df["test_prediction"] = [i for i in test_prediction]
        df.to_csv(save_prediction, index=False)

        print("Save the value of prediction successfully!!")

        # save the model weight
        model_path = os.path.join(save_path, "Model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if self.use_more_gpu:
            torch.save(
                self.net.state_dict(), os.path.join(model_path, str(count) + ".pth")
            )
        else:
            torch.save(
                self.net.state_dict(), os.path.join(model_path, str(count) + ".pth")
            )

        return count

    def reg_calculate(self, true, prediction, features=None):
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
        # rmsle = np.sqrt(metrics.mean_squared_log_error(true, prediction))

        try:
            n = len(true)
            p = features
            r2_adjusted = 1 - ((1 - metrics.r2_score(true, prediction)) * (n - 1)) / (
                n - p - 1
            )
        except:
            # print("mse: {}, rmse: {}, mae: {}, mape: {}, r2: {}, rmsle: {}".format(mse, rmse, mae, mape, r2, rmsle))
            print(
                "if you wanna get the value of r2_adjusted, you can define the number of features, "
                "which is the third parameter."
            )
            # return mse, rmse, mae, mape, r2, rmsle
            return mse, rmse, mae, mape, r2

        # print("mse: {}, rmse: {}, mae: {}, mape: {}, r2: {}, r2_adjusted: {}, rmsle: {}".format(mse, rmse, mae, mape,r2, r2_adjusted, rmsle))
        # return mse, rmse, mae, mape, r2, r2_adjusted, rmsle
        return mse, rmse, mae, mape, r2, r2_adjusted

    # 创建数据生成器
    def create_batch_size(self, X_train, y_train):
        p = np.random.permutation(X_train.shape[0])
        data = X_train[p]
        label = y_train[p]

        batch_size = self.batch_size
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

    # 用于优化模型判断效果
    def fitness(self, evaluationStr="r2"):
        if evaluationStr == "r2":
            return self.r2
        elif evaluationStr == "r2_adjusted":
            return self.r2_adjusted
        elif evaluationStr == "rmsle":
            return self.rmsle
        elif evaluationStr == "mape":
            return self.mape
        elif evaluationStr == "r2_adjusted":
            return self.r2_adjusted
        elif evaluationStr == "mad":
            return self.mad
        elif evaluationStr == "mae":
            return self.mae


if __name__ == "__main__":

    # 导入数据集
    data, label = preprocessing(os.path.join("GeneratedData"))

    # 补全长度：padding
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    data = np.transpose(padded_sequences.numpy(), (0, 1, 2))

    print(data.shape)

    # 定义超参数
    feature_size = 512  # model的特征维度
    nhead = 4  # 注意力机制头数
    num_encoder_layers = 6  # encoder层数
    output_dim = 3  # 输出向量维度（x,y,z
    max_seq_len = data.shape[1]
    input_dim = data.shape[2]

    # 切割数据样本集合测试集
    train_data, test_data, train_label, test_label = train_test_split(
        data, label, test_size=0.2
    )  # 20%测试集；80%训练集
    print([len(train_data), len(train_label), len(test_data), len(test_label)])

    # 初始化模型
    # model = TransAm(input_dim, d_model, nhead, num_encoder_layers, output_dim)
    model = TransAm(
        input_dim=input_dim,
        feature_size=feature_size,
        output_dim=output_dim,
        num_layers=num_encoder_layers,
        dropout=0.5,
        nhead=nhead,
        max_len=max_seq_len,
    )

    grt = General_Regression_Training_3d(
        model,
        learning_rate=[1e-3, 1e-6, 1e-8],
        batch_size=4,
        use_more_gpu=True,
        weight_decay=1e-3,
        device=0,
        save_path="transformer_Result",
        epoch=2000,
    )

    grt.fit(train_data, train_label, test_data, test_label)


# # 对特征操作 使特征为双数
# train_data_Double = []
# for line in train_data:
#     tempList = []
#     for l in line:
#         tempList.extend([l,l])
#     X_train_Double.append([np.array(tempList),np.array(tempList)])

# X_train_Double = np.array(X_train_Double)
# print(X_train_Double.shape)

# X_test_Double = []
# for line in x_test:
#     tempList = []
#     for l in line:
#         tempList.extend([l,l])
#     X_test_Double.append([np.array(tempList),np.array(tempList)])

# X_test_Double = np.array(X_test_Double)

# print("X_train_Double.shape:",X_train_Double.shape,"X_test_Double.shape:",X_test_Double.shape)
# print(X_train_Double[0,0,:],' ',X_train_Double[0,1,:])

# model = TransAm(feature_size=26,num_layers=1,dropout=0.5)
# grt = General_Regression_Training_3d(model,learning_rate = [1e-3,1e-6,1e-8],batch_size = 512,use_more_gpu = True,weight_decay=1e-3, device=0 ,save_path='transformer_Result',epoch = 20000)

# grt.fit(X_train_Double, y_train, X_test_Double, y_true )
