"""
The main training loop
All the methods are implemented in this project
Including Baseline (Ordinary Non-dimensional Equations), InnerNormalization (Only Preprocess the input variables),
and Normalization (Normalize both input and output).

This code uses wandb to grid search the different hyperparameters for comparing
see https://wandb.ai/site/ for details
The login key is covered, you will need your own wandb login key to run this code
wandb.login(key="123456789")
Cheers!

主要训练循环
所有方法都在此项目中实现
包括基线方法（普通无量纲方程）、内部归一化（仅对输入变量进行预处理）
以及归一化（对输入和输出同时进行归一化）。

此代码使用 wandb 进行网格搜索不同的超参数以进行比较，
详情请参阅 https://wandb.ai/site/

登录密钥已被隐藏，您需要使用自己的 wandb 登录密钥来运行此代码。
wandb.login(key="123456789")
加油！
"""
import wandb
from utilities import *
from argparse import Namespace
from validation_func import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")
# Online Running 在线运行
wandb.login(key="")

# Offline Running 离线运行
# os.environ["WANDB_MODE"] = "offline"
Train_Config = Namespace(
    # fixed parameters
    dimension=2 + 1,
    outer_epoch=3000,
    save_interval=100,
    number_eqa=1000000,
    # variable parameters
    optimizer='adam',
    scheduler='exp',
    batch_size=10000,
    learning_rate=1e-3,
    hidden_layers=10,
    layer_neurons=32,
    weight_of_data=1,
    weight_of_eqa=1,
    debug_key=1,
    data_path='./cylinder_Re3900_36points_100snaps.mat',
    real_path='./cylinder_Re3900_ke_all_100snaps.mat',  # 真实参考数据
    evaluate_path='./RL2.csv',
    loss_path='./loss.csv',
    Re=3900,
    dataset='Re123456',
    preprocess='this_is_intentional_mess',
    repeat_exp='Rep1',
    # wandb
    project_name='batch_size_compare'
)

Sweep_Config = {
    'method': 'grid',  # gridsearch, search every combination
    'name': 'parameter_search',

    'metric': {'goal': 'minimize',
               'name': 'RL2_all_epoch'},

    'parameters': {
        'optimizer': {'values': ['adam']},
        'scheduler': {'values': ['exp']},
        'batch_size': {'values': [2048, 8192, 32768]},
        'learning_rate': {'values': [1e-3]},
        'hidden_layers': {'values': [10]},
        'layer_neurons': {'values': [32]},
        'weight_of_data': {'values': [1]},
        'weight_of_eqa': {'values': [1]},
        'dataset': {'values': ['Re48000000']},
        'preprocess': {'values': ['Baseline', 'InnerNorm', 'Normalization']},
        'repeat_exp': {'values': ['Rep1', 'Rep2', 'Rep3']}
    }
}


def config_case_file(case):
    if case == "Re3900":
        data_path = './cylinder_Re3900_36points_100snaps.mat'  # training data 训练数据
        real_path = './cylinder_Re3900_ke_all_100snaps.mat'  # reference data 参考数据
        Re = 3900
    elif case == 'Re2000':
        data_path = './decaying_turbulence_36points_100snaps.mat'  # training data 训练数据
        real_path = './decaying_turbulence_all_100snaps.mat'  # reference data 参考数据
        Re = 2000
    elif case == 'Re48000000':
        data_path = './windfield_Re48000000_sparse_100steps.mat'  # training data 训练数据
        real_path = './windfield_Re48000000_all_100steps.mat'  # reference data 参考数据
        Re = 48000000
    else:
        data_path = './Re10000_kw_36points.mat'  # training data 训练数据
        real_path = './Re10000_kw_all_data.mat'  # reference data 参考数据
        Re = 10000
    return data_path, real_path, Re


# determines the preprocess method 确定使用何种预处理方法
def config_preprocess_method(methodology):
    function_mapping = {
        "Baseline": loop_Baseline,
        "InnerNorm": loop_InnerNorm,
        "Normalization": loop_Normalization
    }
    selected_function = function_mapping.get(methodology)
    if selected_function is None:
        print("Invalid method name. Exiting program.")
        exit()
    return selected_function


def config_parameters(default_config, wandb_sweep_config):
    # pass hyperparameters
    # 传递超参数
    assigned_train_config = default_config
    for key, value in wandb_sweep_config.items():
        if key in assigned_train_config.__dict__:
            setattr(assigned_train_config, key, value)
    case = assigned_train_config.dataset  # train different cases 训练不同算例
    # assign directory according to different cases
    # 根据case不同指定路径
    assigned_train_config.data_path, assigned_train_config.real_path, assigned_train_config.Re = config_case_file(case)
    assigned_train_config.write_path = 'write_sweep/{}_Case_{}_Preprocess_{}_Layer_{}_Neuron_{}_BatchSize_{}_Scheduler_{}' \
        .format(assigned_train_config.repeat_exp, case, assigned_train_config.preprocess,
                assigned_train_config.hidden_layers,
                assigned_train_config.layer_neurons,
                assigned_train_config.batch_size, assigned_train_config.scheduler)
    if not os.path.exists(assigned_train_config.write_path):
        os.makedirs(assigned_train_config.write_path)
    assigned_train_config.evaluate_path = assigned_train_config.write_path + '/RL2.csv'
    assigned_train_config.loss_path = assigned_train_config.write_path + '/loss.csv'
    return assigned_train_config


def train_sweep():
    run = wandb.init(reinit=True, project=Train_Config.project_name, entity="ai_science")
    config = run.config
    parameter_config = config_parameters(Train_Config, config)
    preprocess = parameter_config.preprocess
    loop_function = config_preprocess_method(preprocess)
    loop_function(parameter_config, device)
    print("One Sweep succeeded, you are niubi")
    return


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=Sweep_Config, project=Train_Config.project_name)
    # sweep_id = 'zz7gsbh7'
    print("sweep_id is ", sweep_id)
    # run the sweep
    wandb.agent(sweep_id=sweep_id, function=train_sweep, project=Train_Config.project_name)
