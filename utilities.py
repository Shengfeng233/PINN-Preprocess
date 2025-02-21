"""
core functions
核心函数
"""
import time
import wandb
from validation_func import *


class DataRange:
    def __init__(self):
        self.mean_x, self.mean_y, self.mean_t, self.mean_u, self.mean_v, self.mean_p = 0, 0, 0, 0, 0, 0
        self.std_x, self.std_y, self.std_t, self.std_u, self.std_v, self.std_p = 1, 1, 1, 1, 1, 1


def load_data_feature(filename):
    data_mat = scipy.io.loadmat(filename)
    stack = data_mat['stack']  # N*4 (x,y,u,v)
    x = stack[:, 0].reshape(-1, 1)
    y = stack[:, 1].reshape(-1, 1)
    t = stack[:, 2].reshape(-1, 1)
    temp = np.concatenate((x, y, t), 1)
    data_mean = np.mean(temp, axis=0).reshape(1, -1)
    data_std = np.std(temp, axis=0).reshape(1, -1)
    return data_mean, data_std


# load data points
# 网络加载数据点
def load_data_points(filename):
    data_mat = scipy.io.loadmat(filename)
    stack = data_mat['stack']  # N*4 (x,y,u,v)
    x = stack[:, 0].reshape(-1, 1)
    y = stack[:, 1].reshape(-1, 1)
    t = stack[:, 2].reshape(-1, 1)
    u = stack[:, 3].reshape(-1, 1)
    v = stack[:, 4].reshape(-1, 1)
    p = stack[:, 5].reshape(-1, 1)
    low_bound = np.array([np.min(x), np.min(y), np.min(t)]).reshape(1, -1)
    up_bound = np.array([np.max(x), np.max(y), np.max(t)]).reshape(1, -1)
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.min(temp, 0)
    feature_mat[1, :] = np.max(temp, 0)
    x_ts = torch.tensor(x, dtype=torch.float32)
    y_ts = torch.tensor(y, dtype=torch.float32)
    t_ts = torch.tensor(t, dtype=torch.float32)
    u_ts = torch.tensor(u, dtype=torch.float32)
    v_ts = torch.tensor(v, dtype=torch.float32)
    p_ts = torch.tensor(p, dtype=torch.float32)
    return x_ts, y_ts, t_ts, u_ts, v_ts, p_ts, low_bound, up_bound


# load data points, for normalization method
# 网络加载数据点, Normalization 方法
def load_data_points_Normalization(filename):
    data_mat = scipy.io.loadmat(filename)
    stack = data_mat['stack']  # N*4 (x,y,u,v)
    x = stack[:, 0].reshape(-1, 1)
    y = stack[:, 1].reshape(-1, 1)
    t = stack[:, 2].reshape(-1, 1)
    u = stack[:, 3].reshape(-1, 1)
    v = stack[:, 4].reshape(-1, 1)
    p = stack[:, 5].reshape(-1, 1)
    # Normalization
    # 归一化处理
    mean_x, mean_y, mean_t, mean_u, mean_v, mean_p = np.mean(x), np.mean(y), np.mean(t), np.mean(u), np.mean(
        v), np.mean(p)
    std_x, std_y, std_t, std_u, std_v, std_p = np.std(x), np.std(y), np.std(t), np.std(u), np.std(v), np.std(p)

    # record mean and standard deviation values
    # 记录均值和标准差
    data_range = DataRange()
    data_range.mean_x, data_range.mean_y, data_range.mean_t, data_range.mean_u, data_range.mean_v, data_range.mean_p = mean_x, mean_y, mean_t, mean_u, mean_v, mean_p
    data_range.std_x, data_range.std_y, data_range.std_t, data_range.std_u, data_range.std_v, data_range.std_p = std_x, std_y, std_t, std_u, std_v, std_p

    # Transfer to torch tensor
    # 转化为Tensor
    x = (x - mean_x) / std_x
    y = (y - mean_y) / std_y
    t = (t - mean_t) / std_t
    u = (u - mean_u) / std_u
    v = (v - mean_v) / std_v
    p = (p - mean_p) / std_p
    low_bound = np.array([np.min(x), np.min(y), np.min(t)]).reshape(1, -1)
    up_bound = np.array([np.max(x), np.max(y), np.max(t)]).reshape(1, -1)
    temp = np.concatenate((x, y, t, u, v, p), 1)
    x_ts = torch.tensor(x, dtype=torch.float32)
    y_ts = torch.tensor(y, dtype=torch.float32)
    t_ts = torch.tensor(t, dtype=torch.float32)
    u_ts = torch.tensor(u, dtype=torch.float32)
    v_ts = torch.tensor(v, dtype=torch.float32)
    p_ts = torch.tensor(p, dtype=torch.float32)
    return x_ts, y_ts, t_ts, u_ts, v_ts, p_ts, low_bound, up_bound, data_range


# load equation points
# 加载方程点——拉丁超立方
def load_equation_points_lhs(low_bound, up_bound, dimension, points):
    eqa_xyzt = low_bound + (up_bound - low_bound) * lhs(dimension, points)
    Eqa_points = torch.from_numpy(eqa_xyzt).float()
    Eqa_points = Eqa_points[torch.randperm(Eqa_points.size(0))]
    return Eqa_points


# split batch and automatically fill batch size
# batch 划分与自动填充
def batch_split(Set, iter_num, dim=0):
    batches = torch.chunk(Set, iter_num, dim=dim)
    # 自动填充
    num_of_batches = len(batches)
    if num_of_batches == 1:
        batches = batches * iter_num
        return batches
    if num_of_batches < iter_num:
        for i in range(iter_num - num_of_batches):
            index = i % num_of_batches
            add_tuple = batches[-(index + 2):-(index + 1)]
            batches = batches + add_tuple
        return batches
    else:
        return batches


# preloading before training
# 训练前预加载
# 数据划分batch
def pre_train_loading(filename_data, dimension, N_eqa, batch_size):
    # load data points(only once)
    # 加载真实数据点(仅一次)
    x_sub_ts, y_sub_ts, t_sub_ts, u_sub_ts, v_sub_ts, p_sub_ts, low_bound, up_bound = load_data_points(filename_data)
    if x_sub_ts.shape[0] > 0:
        data_sub = torch.cat([x_sub_ts, y_sub_ts, t_sub_ts, u_sub_ts, v_sub_ts, p_sub_ts], 1)
        true_dataset = data_sub[torch.randperm(data_sub.size(0))]  # permutation 乱序
    else:
        true_dataset = None
    # load collocation points(only once)
    # 加载方程点(仅一次)
    eqa_points = load_equation_points_lhs(low_bound, up_bound, dimension, N_eqa)
    eqa_points_batches = torch.split(eqa_points, batch_size, dim=0)
    iter_num = len(eqa_points_batches)
    true_dataset_batches = batch_split(true_dataset, iter_num)
    return true_dataset, eqa_points_batches, iter_num, low_bound, up_bound


def pre_train_loading_Normalization(filename_data, dimension, N_eqa, batch_size):
    # load data points(only once)
    # 加载真实数据点(仅一次)
    x_sub_ts, y_sub_ts, t_sub_ts, u_sub_ts, v_sub_ts, p_sub_ts, low_bound, up_bound, data_range = load_data_points_Normalization(
        filename_data)
    if x_sub_ts.shape[0] > 0:
        data_sub = torch.cat([x_sub_ts, y_sub_ts, t_sub_ts, u_sub_ts, v_sub_ts, p_sub_ts], 1)
        true_dataset = data_sub[torch.randperm(data_sub.size(0))]  # 乱序
    else:
        true_dataset = None
    # load collocation points(only once)
    # 加载方程点(仅一次)
    eqa_points = load_equation_points_lhs(low_bound, up_bound, dimension, N_eqa)
    eqa_points_batches = torch.split(eqa_points, batch_size, dim=0)
    iter_num = len(eqa_points_batches)
    true_dataset_batches = batch_split(true_dataset, iter_num)
    return true_dataset, eqa_points_batches, iter_num, low_bound, up_bound, data_range


# Baseline Method
# train data points, collocation points--with batch training
# 同时训练数据点，方程点———有batch
def train_data_Baseline(pinn_example, optimizer_all, scheduler_all, iter_num, true_dataset,
                        Eqa_points_batches, Re, weight_data, weight_eqa, EPOCH, debug_key, device):
    loss_sum = np.array([0.0]).reshape(1, 1)
    loss_data = np.array([0.0]).reshape(1, 1)
    loss_eqa = np.array([0.0]).reshape(1, 1)
    x_train = true_dataset[:, 0].reshape(-1, 1).requires_grad_(True).to(device)
    y_train = true_dataset[:, 1].reshape(-1, 1).requires_grad_(True).to(device)
    t_train = true_dataset[:, 2].reshape(-1, 1).requires_grad_(True).to(device)
    u_train = true_dataset[:, 3].reshape(-1, 1).to(device)
    v_train = true_dataset[:, 4].reshape(-1, 1).to(device)
    p_train = true_dataset[:, 5].reshape(-1, 1).to(device)
    for batch_iter in range(iter_num):
        optimizer_all.zero_grad()
        # 数据划分batch，batch数和方程点的batch数目一致
        x_eqa = Eqa_points_batches[batch_iter][:, 0].reshape(-1, 1).requires_grad_(True).to(device)
        y_eqa = Eqa_points_batches[batch_iter][:, 1].reshape(-1, 1).requires_grad_(True).to(device)
        t_eqa = Eqa_points_batches[batch_iter][:, 2].reshape(-1, 1).requires_grad_(True).to(device)
        mse_data = pinn_example.data_mse(x_train, y_train, t_train, u_train, v_train, p_train)
        mse_equation = pinn_example.equation_mse_dimensionless(x_eqa, y_eqa, t_eqa, Re)
        # calculate loss
        # 计算损失函数
        loss = weight_data * mse_data + weight_eqa * mse_equation
        loss.backward()
        optimizer_all.step()
        with torch.autograd.no_grad():
            loss_sum = loss.cpu().data.numpy()
            loss_data = mse_data.cpu().data.numpy()
            loss_eqa = mse_equation.cpu().data.numpy()
            # print status
            # 输出状态
            if (batch_iter + 1) % iter_num == 0 and debug_key == 1:
                print("EPOCH:", (EPOCH + 1), "  inner_iter:", batch_iter + 1, " Training-data Loss:",
                      round(float(loss.data), 8))
    scheduler_all.step()
    return loss_sum, loss_data, loss_eqa


# InnerNorm Method
# train data points, collocation points--with batch training
# 同时训练数据点，方程点———有batch
def train_data_InnerNorm(pinn_example, optimizer_all, scheduler_all, iter_num, true_dataset,
                         Eqa_points_batches, Re, weight_data, weight_eqa, EPOCH, debug_key, device):
    loss_sum = np.array([0.0]).reshape(1, 1)
    loss_data = np.array([0.0]).reshape(1, 1)
    loss_eqa = np.array([0.0]).reshape(1, 1)
    x_train = true_dataset[:, 0].reshape(-1, 1).requires_grad_(True).to(device)
    y_train = true_dataset[:, 1].reshape(-1, 1).requires_grad_(True).to(device)
    t_train = true_dataset[:, 2].reshape(-1, 1).requires_grad_(True).to(device)
    u_train = true_dataset[:, 3].reshape(-1, 1).to(device)
    v_train = true_dataset[:, 4].reshape(-1, 1).to(device)
    p_train = true_dataset[:, 5].reshape(-1, 1).to(device)
    for batch_iter in range(iter_num):
        optimizer_all.zero_grad()
        x_eqa = Eqa_points_batches[batch_iter][:, 0].reshape(-1, 1).requires_grad_(True).to(device)
        y_eqa = Eqa_points_batches[batch_iter][:, 1].reshape(-1, 1).requires_grad_(True).to(device)
        t_eqa = Eqa_points_batches[batch_iter][:, 2].reshape(-1, 1).requires_grad_(True).to(device)
        mse_data = pinn_example.data_mse_inner_norm(x_train, y_train, t_train, u_train, v_train, p_train)
        mse_equation = pinn_example.equation_mse_dimensionless_inner_norm(x_eqa, y_eqa, t_eqa, Re)
        # calculate loss
        # 计算损失函数
        loss = weight_data * mse_data + weight_eqa * mse_equation
        loss.backward()
        optimizer_all.step()
        with torch.autograd.no_grad():
            loss_sum = loss.cpu().data.numpy()
            loss_data = mse_data.cpu().data.numpy()
            loss_eqa = mse_equation.cpu().data.numpy()
            # print status
            # 输出状态
            if (batch_iter + 1) % iter_num == 0 and debug_key == 1:
                print("EPOCH:", (EPOCH + 1), "  inner_iter:", batch_iter + 1, " Training-data Loss:",
                      round(float(loss.data), 8))
    scheduler_all.step()
    return loss_sum, loss_data, loss_eqa


# Normalization Method
# train data points, collocation points--with batch training
# 同时训练数据点，方程点———有batch
def train_data_Normalization(pinn_example, optimizer_all, scheduler_all, iter_num, true_dataset, Eqa_points_batches,
                             data_range, Re, weight_data, weight_eqa, EPOCH, debug_key, device):
    loss_sum = np.array([0.0]).reshape(1, 1)
    loss_data = np.array([0.0]).reshape(1, 1)
    loss_eqa = np.array([0.0]).reshape(1, 1)
    x_train = true_dataset[:, 0].reshape(-1, 1).requires_grad_(True).to(device)
    y_train = true_dataset[:, 1].reshape(-1, 1).requires_grad_(True).to(device)
    t_train = true_dataset[:, 2].reshape(-1, 1).requires_grad_(True).to(device)
    u_train = true_dataset[:, 3].reshape(-1, 1).to(device)
    v_train = true_dataset[:, 4].reshape(-1, 1).to(device)
    p_train = true_dataset[:, 5].reshape(-1, 1).to(device)
    for batch_iter in range(iter_num):
        optimizer_all.zero_grad()
        # 数据划分batch，batch数和方程点的batch数目一致
        x_eqa = Eqa_points_batches[batch_iter][:, 0].reshape(-1, 1).requires_grad_(True).to(device)
        y_eqa = Eqa_points_batches[batch_iter][:, 1].reshape(-1, 1).requires_grad_(True).to(device)
        t_eqa = Eqa_points_batches[batch_iter][:, 2].reshape(-1, 1).requires_grad_(True).to(device)
        mse_data = pinn_example.data_mse(x_train, y_train, t_train, u_train, v_train, p_train)
        mse_equation = pinn_example.equation_mse_normal_dimensionless(x_eqa, y_eqa, t_eqa, Re, data_range)
        # calculate loss
        # 计算损失函数
        loss = weight_data * mse_data + weight_eqa * mse_equation
        loss.backward()
        optimizer_all.step()
        with torch.autograd.no_grad():
            loss_sum = loss.cpu().data.numpy()
            loss_data = mse_data.cpu().data.numpy()
            loss_eqa = mse_equation.cpu().data.numpy()
            # print status
            # 输出状态
            if (batch_iter + 1) % iter_num == 0 and debug_key == 1:
                print("EPOCH:", (EPOCH + 1), "  inner_iter:", batch_iter + 1, " Training-data Loss:",
                      round(float(loss.data), 8))
    scheduler_all.step()
    return loss_sum, loss_data, loss_eqa


# record loss
# 记录loss
def record_loss_local(loss_sum, loss_data, loss_eqa, filename_loss):
    loss_sum_value = loss_sum.reshape(1, 1)
    loss_data_value = loss_data.reshape(1, 1)
    loss_eqa_value = loss_eqa.reshape(1, 1)
    loss_set = np.concatenate((loss_sum_value, loss_data_value, loss_eqa_value), 1).reshape(1, -1)
    loss_save = pd.DataFrame(loss_set)
    loss_save.to_csv(filename_loss, index=False, header=False, mode='a')
    return loss_set


def build_optimizer(network, optimizer_name, scheduler_name, learning_rate):
    # default 默认优化器
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    # 超参数搜索优化器
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    if scheduler_name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    elif scheduler_name == "fix":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    return optimizer, scheduler


def loop_Baseline(parameter_config, device):
    # unpack important parameters
    # 重要参数
    dimension = parameter_config.dimension
    save_interval = parameter_config.save_interval  # save interval 保存模型周期
    outer_epochs = parameter_config.outer_epoch  # training epochs 训练周期数
    number_eqa = parameter_config.number_eqa
    debug_key = parameter_config.debug_key
    layer_mat = [dimension] + parameter_config.hidden_layers * [parameter_config.layer_neurons] + [3]
    learning_rate = parameter_config.learning_rate  # 学习率 learning rate
    batch_size = parameter_config.batch_size
    weight_of_data = parameter_config.weight_of_data
    weight_of_eqa = parameter_config.weight_of_eqa
    optimizer_name = parameter_config.optimizer
    scheduler_name = parameter_config.scheduler
    data_path = parameter_config.data_path
    real_path = parameter_config.real_path
    Re = parameter_config.Re
    write_path = parameter_config.write_path
    evaluate_path = parameter_config.evaluate_path
    loss_path = parameter_config.loss_path

    True_dataset_batches, Eqa_points_batches, iter_num, low_bound, up_bound = pre_train_loading(
        data_path,
        dimension,
        number_eqa,
        batch_size, )
    data_mean, data_std = load_data_feature(data_path)
    pinn_net = PINN_Net(layer_mat, data_mean, data_std, device)
    pinn_net = pinn_net.to(device)

    # 优化器和学习率衰减设置- optimizer and learning rate schedule
    optimizer_all, scheduler_all = build_optimizer(pinn_net, optimizer_name, scheduler_name, learning_rate)

    start = time.time()

    # 训练主循环 main loop
    for EPOCH in range(outer_epochs):

        # Training, every iteration, all data is trained once
        # 对数据点，方程点进行同时训练 train----每一个iter中训练全部数据
        loss_sum, loss_data, loss_eqa = train_data_Baseline(pinn_net, optimizer_all, scheduler_all,
                                                            iter_num, True_dataset_batches, Eqa_points_batches,
                                                            Re, weight_of_data, weight_of_eqa, EPOCH, debug_key,
                                                            device)

        # loss记录 record loss
        loss_set = record_loss_local(loss_sum, loss_data, loss_eqa, loss_path)  # 记录子网络loss

        # 每隔固定Epoch保存模型 save model at every save_interval epoch
        if ((EPOCH + 1) % save_interval == 0) | (EPOCH == 0):
            # dir_name = write_path + '/step' + str(EPOCH + 1)
            # os.makedirs(dir_name, exist_ok=True)
            torch.save(pinn_net.state_dict(), write_path + '/NS_model_train_{}.pt'.format(EPOCH + 1))
            print(f'Model saved at step {EPOCH + 1}.')
            valid_u, valid_v, valid_p = validation_Baseline(pinn_net, real_path, evaluate_path, device)
            wandb.log({"ReL2_u": valid_u, "ReL2_v": valid_v, "ReL2_p": valid_p})
            print(f'Model evaluated at step {EPOCH + 1}.')

    end = time.time()
    print("Time used: ", end - start)
    return


def loop_InnerNorm(parameter_config, device):
    # unpack important parameters
    # 重要参数
    dimension = parameter_config.dimension
    save_interval = parameter_config.save_interval   # save interval 保存模型周期
    outer_epochs = parameter_config.outer_epoch  # training epochs 训练周期数
    number_eqa = parameter_config.number_eqa
    debug_key = parameter_config.debug_key
    layer_mat = [dimension] + parameter_config.hidden_layers * [parameter_config.layer_neurons] + [3]
    learning_rate = parameter_config.learning_rate  # 学习率 learning rate
    batch_size = parameter_config.batch_size
    weight_of_data = parameter_config.weight_of_data
    weight_of_eqa = parameter_config.weight_of_eqa
    optimizer_name = parameter_config.optimizer
    scheduler_name = parameter_config.scheduler
    data_path = parameter_config.data_path
    real_path = parameter_config.real_path
    Re = parameter_config.Re
    write_path = parameter_config.write_path
    evaluate_path = parameter_config.evaluate_path
    loss_path = parameter_config.loss_path

    True_dataset_batches, Eqa_points_batches, iter_num, low_bound, up_bound = pre_train_loading(
        data_path,
        dimension,
        number_eqa,
        batch_size, )
    data_mean, data_std = load_data_feature(data_path)
    pinn_net = PINN_Net(layer_mat, data_mean, data_std, device)
    pinn_net = pinn_net.to(device)

    # 优化器和学习率衰减设置- optimizer and learning rate schedule
    optimizer_all, scheduler_all = build_optimizer(pinn_net, optimizer_name, scheduler_name, learning_rate)

    start = time.time()

    # 训练主循环 main loop
    for EPOCH in range(outer_epochs):

        # 对数据点，方程点进行同时训练 train----每一个iter中训练全部数据
        loss_sum, loss_data, loss_eqa = train_data_InnerNorm(pinn_net, optimizer_all,
                                                             scheduler_all,
                                                             iter_num, True_dataset_batches,
                                                             Eqa_points_batches, Re, weight_of_data,
                                                             weight_of_eqa, EPOCH,
                                                             debug_key, device)

        # loss记录 record loss
        loss_set = record_loss_local(loss_sum, loss_data, loss_eqa, loss_path)  # 记录子网络loss

        # 每隔固定Epoch保存模型 save model at every save_interval epoch
        if ((EPOCH + 1) % save_interval == 0) | (EPOCH == 0):
            # dir_name = write_path + '/step' + str(EPOCH + 1)
            # os.makedirs(dir_name, exist_ok=True)
            torch.save(pinn_net.state_dict(), write_path + '/NS_model_train_{}.pt'.format(EPOCH + 1))
            print(f'Model saved at step {EPOCH + 1}.')
            valid_u, valid_v, valid_p = validation_InnerNorm(pinn_net, real_path, evaluate_path, device)
            wandb.log({"ReL2_u": valid_u, "ReL2_v": valid_v, "ReL2_p": valid_p})
            print(f'Model evaluated at step {EPOCH + 1}.')

    end = time.time()
    print("Time used: ", end - start)
    return


def loop_Normalization(parameter_config, device):
    # unpack important parameters
    # 重要参数
    dimension = parameter_config.dimension
    save_interval = parameter_config.save_interval  # save interval 保存模型周期
    outer_epochs = parameter_config.outer_epoch  # training epochs 训练周期数
    number_eqa = parameter_config.number_eqa
    debug_key = parameter_config.debug_key
    layer_mat = [dimension] + parameter_config.hidden_layers * [parameter_config.layer_neurons] + [3]
    learning_rate = parameter_config.learning_rate  # 学习率 learning rate
    batch_size = parameter_config.batch_size
    weight_of_data = parameter_config.weight_of_data
    weight_of_eqa = parameter_config.weight_of_eqa
    optimizer_name = parameter_config.optimizer
    scheduler_name = parameter_config.scheduler
    data_path = parameter_config.data_path
    real_path = parameter_config.real_path
    Re = parameter_config.Re
    write_path = parameter_config.write_path
    evaluate_path = parameter_config.evaluate_path
    loss_path = parameter_config.loss_path
    # pretraining-loading data
    # loading data points
    # loading collocation points
    # only load once
    # 训练前加载工作(仅加载一次)
    # 加载数据点，加载方程点
    True_dataset_batches, Eqa_points_batches, iter_num, low_bound, up_bound, data_range = pre_train_loading_Normalization(
        data_path,
        dimension,
        number_eqa,
        batch_size)
    data_mean, data_std = load_data_feature(data_path)
    pinn_net = PINN_Net(layer_mat, data_mean, data_std, device)
    pinn_net = pinn_net.to(device)
    # 优化器和学习率衰减设置- optimizer and learning rate schedule
    optimizer_all, scheduler_all = build_optimizer(pinn_net, optimizer_name, scheduler_name, learning_rate)

    start = time.time()
    # 训练主循环 main loop
    for EPOCH in range(outer_epochs):

        # Training, every iteration, all data is trained once
        # 对数据点，方程点进行同时训练 train----每一个iter中训练全部数据
        loss_sum, loss_data, loss_eqa = train_data_Normalization(pinn_net, optimizer_all, scheduler_all, iter_num,
                                                                 True_dataset_batches, Eqa_points_batches, data_range,
                                                                 Re, weight_of_data, weight_of_eqa, EPOCH, debug_key,
                                                                 device)
        # loss记录 record loss
        loss_set = record_loss_local(loss_sum, loss_data, loss_eqa, loss_path)  # 记录子网络loss

        # 每隔固定Epoch保存模型 save model at every save_interval epoch
        if ((EPOCH + 1) % save_interval == 0) | (EPOCH == 0):
            # dir_name = write_path + '/step' + str(EPOCH + 1)
            # os.makedirs(dir_name, exist_ok=True)
            torch.save(pinn_net.state_dict(), write_path + '/NS_model_train_{}.pt'.format(EPOCH + 1))
            print(f'Model saved at step {EPOCH + 1}.')
            valid_u, valid_v, valid_p = validation_Normalization(pinn_net, real_path, data_range, evaluate_path, device)
            wandb.log({"ReL2_u": valid_u, "ReL2_v": valid_v, "ReL2_p": valid_p})
            print(f'Model evaluated at step {EPOCH + 1}.')

    end = time.time()
    print("Time used: ", end - start)
    return
