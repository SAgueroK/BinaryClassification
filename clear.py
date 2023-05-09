import config
import utils

# 清空所有模型和log文件，不知道为啥log文件清空函数 不能放在train里面，会报文件正在被另一个程序使用的错误
if __name__ == '__main__':
    utils.del_files(config.HP.model_save_dir)
    utils.del_files(config.HP.log_dir_eval)
    utils.del_files(config.HP.log_dir_train)
