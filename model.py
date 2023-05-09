import numpy
import numpy as np
import torch
import torch.nn as nn
import config


class Model(nn.Module):
    def __init__(self, in_feature=config.Hyperparameter.in_feature, hidden_dim=config.Hyperparameter.hidden_dim
                 , output_size=config.Hyperparameter.output_size, layer_num=config.Hyperparameter.lstm_layer_num,
                 batch_size=config.Hyperparameter.batch_size, time_step=config.Hyperparameter.time_step):
        """
        LSTM二分类任务
        :param in_feature: 输入数据的维度
        :param hidden_dim:隐层维度
        :param layer_num: lstm的层数
        :param output_size: 输出的个数
        :param batch_size: 取样个数
        """
        super().__init__()
        self.in_feature = in_feature
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.output_size = output_size
        self.batch_size = batch_size
        self.time_step = time_step

        self.lstm = nn.LSTM(input_size=in_feature, hidden_size=hidden_dim, num_layers=layer_num)
        self.linear = nn.Linear(hidden_dim * layer_num, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        # 先将输入从 batch_size * time_step * in_feature 转为 time_step * batch_size * in_feature
        input_x = torch.tensor(input_x.numpy().transpose((1, 0, 2)), dtype=torch.float)
        hid = torch.randn(self.layer_num, self.batch_size, self.hidden_dim, dtype=torch.float)
        cell = torch.randn(self.layer_num, self.batch_size, self.hidden_dim, dtype=torch.float)
        lstm_out, (h_n, h_c) = self.lstm(input_x, (hid, cell))
        # print("h_c的输出维度", h_c.shape)
        liner_input = torch.clone(h_c.view(self.batch_size, -1)).detach().float()
        # print("liner_input的输出维度", liner_input.shape)
        linear_out = self.linear(liner_input)  # =self.linear(lstm_out[:, -1, :])
        # print("liner的输出维度",linear_out.shape)
        predictions = self.sigmoid(linear_out)
        return predictions
