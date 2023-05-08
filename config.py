# classification config

# 超参配置
# yaml
class Hyperparameter:
    # ################################################################
    #                             Data
    # ################################################################
    device = 'cpu'  # cuda
    data_dir_origin_true = './datasets/origin/true'  # 用户的源数据地址
    data_dir_origin_false = './datasets/origin/false'  # 非用户的源数据地址
    data_dir_train_true = './datasets/train/true'  # 用户训练数据地址
    data_dir_train_false = './datasets/train/false'  # 非用户训练数据地址
    data_dir_dev_true = './datasets/dev/true'  # 用户验证数据地址
    data_dir_dev_false = './datasets/dev/false'  # 非用户验证数据地址
    data_dir_test_true = './datasets/test/true'  # 用户测试数据地址
    data_dir_test_false = './datasets/test/false'  # 非用户测试数据地址
    log_dir = './log'  # 日志地址
    model_save_dir = './model_save'   # 模型保存地址
    in_feature = 5  # 输入维度
    time_step = 300  # 时序步
    hidden_dim = 128  # 隐含层输出维度
    lstm_layer_num = 2  # LSTM层数
    output_size = 1  # 输出维度
    seed = 1234  # random seed

    # ################################################################
    #                             Model Structure
    # ################################################################
    # ################################################################
    #                             Experiment
    # ################################################################
    batch_size = 2
    init_lr = 1e-3
    epochs = 10
    verbose_step = 10  # 验证频率
    save_step = 20  # 模型保存频率


HP = Hyperparameter()
