import os
import shutil
import config
import torch
import numpy as np


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


def copyfile(origin_file, destination_path):  # 复制函数:将origin_file文件复制到destination_path中
    destination_path = destination_path + '/'  # destination_path要加/
    if not os.path.isfile(origin_file):
        print("%s not exist!" % origin_file)
    else:
        f_path, f_name = os.path.split(origin_file)  # 分离文件名和路径
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)  # 创建路径
        shutil.copy(origin_file, destination_path + f_name)  # 复制文件
        print("copy %s -> %s" % (origin_file, destination_path + f_name))


def get_all_file_path(origin_path):  # 获取origin_path下的所有文件，并且返回文件项目相对路径,例：./datasets/origin/false/2023-05-0812.12.47
    origin_files = os.listdir(origin_path)
    for i, file in enumerate(origin_files):
        origin_files[i] = origin_path + '/' + file
    return origin_files


# 数据集初始化，将所有初试文件打乱后按6:2:2分配给训练，验证和测试集
def data_init():
    # 获取训练 测试 验证 文件路径
    filePath_origin_true = config.Hyperparameter.data_dir_origin_true
    filePath_origin_false = config.Hyperparameter.data_dir_origin_false
    filePath_train_true = config.Hyperparameter.data_dir_train_true
    filePath_train_false = config.Hyperparameter.data_dir_train_false
    filePath_dev_true = config.Hyperparameter.data_dir_dev_true
    filePath_dev_false = config.Hyperparameter.data_dir_dev_false
    filePath_test_true = config.Hyperparameter.data_dir_test_true
    filePath_test_false = config.Hyperparameter.data_dir_test_false

    # 先清空文件夹
    del_files(filePath_train_true)
    del_files(filePath_train_false)
    del_files(filePath_dev_true)
    del_files(filePath_dev_false)
    del_files(filePath_test_true)
    del_files(filePath_test_false)

    # 获取所有文件
    fileList_x_true = get_all_file_path(filePath_origin_true)
    fileList_x_false = get_all_file_path(filePath_origin_false)
    # 获取y数据 = 用户数据个数的1 + 非用户个数的0
    fileList_y_all = np.append(np.ones(len(fileList_x_true)), (np.zeros(len(fileList_x_false))))
    # 获取所有用户和非用户源文件的路径
    fileList_x_all = fileList_x_true + fileList_x_false  # 获取所有用户和非用户源文件的路径
    # 创建一个index数据，用来打乱
    indexList = [i for i in range(len(fileList_y_all))]
    # 打乱文件路径index列表
    np.random.shuffle(indexList)
    file_num = len(indexList)  # 文件个数
    # 将所有初试文件打乱后按6:2:2分配给训练，验证和测试集
    fileList_train_index = indexList[:int(file_num * 0.6)]
    fileList_dev_index = indexList[int(file_num * 0.6):int(file_num * 0.8)]
    fileList_test_index = indexList[int(file_num * 0.8):]
    print(indexList)
    print(fileList_x_all)
    # 复制训练文件
    for index in fileList_train_index:
        if fileList_y_all[index] == 1:
            copyfile(fileList_x_all[index], filePath_train_true)
        else:
            copyfile(fileList_x_all[index], filePath_train_false)
    for index in fileList_dev_index:
        if fileList_y_all[index] == 1:
            copyfile(fileList_x_all[index], filePath_dev_true)
        else:
            copyfile(fileList_x_all[index], filePath_dev_false)
    for index in fileList_test_index:
        if fileList_y_all[index] == 1:
            copyfile(fileList_x_all[index], filePath_test_true)
        else:
            copyfile(fileList_x_all[index], filePath_test_false)


# 获取数据集
def get_train_data(path_true, path_false):
    # 获取文件路径
    fileList_true = get_all_file_path(path_true)
    fileList_false = get_all_file_path(path_false)
    # 获取时序步长和输入维度
    time_step = config.Hyperparameter.time_step
    in_feature = config.Hyperparameter.in_feature
    # x_data表示最后的输入集
    data_num = len(fileList_true) + len(fileList_false)
    x_data = np.empty((data_num, time_step, in_feature), dtype=float)
    index = 0
    # 获取数据集
    for file in fileList_true:
        tmp_data = np.loadtxt(file, delimiter=',')
        # 对数据进行裁剪和填充 形成 tmp_data*in_feature 维度
        if time_step > np.size(tmp_data, 0):
            tmp_data = np.append(tmp_data, np.zeros((time_step - np.size(tmp_data, 0), in_feature)), 0)
        else:
            tmp_data = tmp_data[:time_step]
        x_data[index] = tmp_data
        index += 1
    for file in fileList_false:
        tmp_data = np.loadtxt(file, delimiter=',')
        # 对数据进行裁剪和填充 形成 tmp_data*in_feature 维度
        if time_step > np.size(tmp_data, 0):
            tmp_data = np.append(tmp_data, np.zeros((time_step - np.size(tmp_data, 0), in_feature)), 0)
        else:
            tmp_data = tmp_data[:time_step]
        x_data[index] = tmp_data
        index += 1
    # 设定输出集
    y_data = np.append(np.ones(len(fileList_true)), (np.zeros(len(fileList_false))))
    # 将输入输出集都转化为张量
    x_data = torch.tensor(x_data, dtype=torch.float)
    y_data = torch.tensor(y_data, dtype=torch.float)
    return x_data, y_data


if __name__ == '__main__':
    data_init()
