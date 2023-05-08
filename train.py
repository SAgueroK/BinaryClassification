import os

import torch
import torch.nn as nn
import torch.utils.data as Data
from tensorboardX import SummaryWriter

import config
import utils
from model import Model

logger_train = SummaryWriter('./log/train')
logger_eval = SummaryWriter('./log/eval')


# 验证
def evaluate(model, dev_loader, loss_function):
    model.eval()  # 切换验证模式
    sum_loss = 0.
    with torch.no_grad():
        for seq, labels in dev_loader:
            y_pred = model(seq)
            labels = torch.clone(labels.view(config.HP.batch_size, config.HP.output_size)).detach().float()
            loss = loss_function(y_pred, labels)
            sum_loss += loss.item()

    model.train()  # back to training mode
    return sum_loss / len(dev_loader)


# 保存模型
def save_checkpoint(model_, epoch_, optimizer, checkpoint_path):
    save_dict = {
        'epoch': epoch_,
        'model_state_dict': model_.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(save_dict, checkpoint_path)


def train():
    # 加载数训练据
    train_x, train_y = utils.get_train_data(config.HP.data_dir_train_true, config.HP.data_dir_train_false)
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(train_x, train_y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=config.HP.batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )
    # 加载验证数据
    dev_x, dev_y = utils.get_train_data(config.HP.data_dir_dev_true, config.HP.data_dir_dev_false)
    dev_loader = Data.DataLoader(
        dataset=Data.TensorDataset(dev_x, dev_y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=config.HP.batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )
    # 建模三件套：loss，优化，epochs
    model = Model()  # 模型
    loss_function = nn.BCELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    model.train()  # 开始训练
    start_epochs = 0  # 开始的epoch
    epochs = config.HP.epochs  # 结束的epoch
    step = 0  # 步数
    for epoch in range(start_epochs, epochs):
        loss_sum = 0.  # loss总损失值
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)  # 得到输出
            # y_pred = torch.round(y_pred)  # 二分类问题使用四舍五入的方法
            # 将labels转换成1*1的张量
            labels = torch.clone(labels.view(config.HP.batch_size, config.HP.output_size)).detach().float()
            loss = loss_function(y_pred, labels)  # 计算损失
            loss.backward()
            logger_train.add_scalar('Loss', loss, step)  # 将loss存入log中
            optimizer.step()
            loss_sum += loss
            print("损失值：", loss)
            if not step % config.HP.verbose_step:  # 每verbose_step次进行一次验证
                eval_loss = evaluate(model, dev_loader, loss_function)
                logger_eval.add_scalar('Loss', eval_loss, step)  # 将验证loss存入log中
                print("验证损失值：", eval_loss)
            if not step % config.HP.save_step:  # 每save_step次进行一次保存
                model_path = 'model_%d_%d.pth' % (epoch, step)
                save_checkpoint(model, epoch, optimizer, os.path.join('model_save', model_path))
            logger_train.flush()
            logger_eval.flush()
            step += 1


if __name__ == '__main__':
    train()
