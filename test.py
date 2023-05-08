import numpy as np
import torch

import config
import utils
from model import Model
import torch.utils.data as Data


def test(test_path):
    model = Model()
    checkpoint = torch.load(test_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_x, test_y = utils.get_train_data(config.HP.data_dir_test_true, config.HP.data_dir_test_false)
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(test_x, test_y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=config.HP.batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )
    model.eval()
    TP = 0.  # 正样本预测答案正确
    FP = 0.  # 错将负样本预测为正样本
    FN = 0.  # 错将正样本标签预测为负样本
    TN = 0.  # 负样本预测答案正确
    with torch.no_grad():
        for seq, labels in test_loader:
            y_pred = model(seq)
            labels = torch.clone(labels.view(-1)).detach().int()
            y_pred = torch.round(y_pred).int()  # 二分类问题使用四舍五入的方法
            y_pred = torch.clone(y_pred.view(-1)).detach().int()
            print("y_pred:", y_pred)
            print("labels:", labels)
            for i, y in enumerate(y_pred):
                if y == 1:
                    if labels[i] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if labels[i] == 1:
                        FN += 1
                    else:
                        TN += 1
    return TP, FP, FN, TN


if __name__ == '__main__':
    TP, FP, FN, TN = test('./model_save/model_0_0.pth')
    Accuracy = (TP + TN) / (TP + FP + FN + TN)  # 准确率
    Precision = -1  # 精确率
    Recall = -1  # 召回率
    if TP + FP != 0:
        Precision = TP / (TP + FP)
    if TP + FN != 0:
        Recall = TP / (TP + FN)
    print("准确率:", Accuracy)
    print("精确率:", Precision)
    print("召回率:", Recall)
