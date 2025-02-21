"""
validation function
To evaluation the L2 norm error during training
You can also evaluate after training
This is only for convenience
"""
import scipy
from pinn_model import *
import pandas as pd


def load_valid_points(filename):
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
    x_ts = torch.tensor(x, dtype=torch.float32)
    y_ts = torch.tensor(y, dtype=torch.float32)
    t_ts = torch.tensor(t, dtype=torch.float32)
    u_ts = torch.tensor(u, dtype=torch.float32)
    v_ts = torch.tensor(v, dtype=torch.float32)
    p_ts = torch.tensor(p, dtype=torch.float32)
    return x_ts, y_ts, t_ts, u_ts, v_ts, p_ts, low_bound, up_bound


def load_valid_points_Normalization(data_range, filename):
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
    mean_x, mean_y, mean_t, mean_u, mean_v, mean_p = data_range.mean_x, data_range.mean_y, data_range.mean_t, data_range.mean_u, data_range.mean_v, data_range.mean_p
    std_x, std_y, std_t, std_u, std_v, std_p = data_range.std_x, data_range.std_y, data_range.std_t, data_range.std_u, data_range.std_v, data_range.std_p
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
    return x_ts, y_ts, t_ts, u_ts, v_ts, p_ts, low_bound, up_bound


def compute_L2_norm(filename_raw_data, pinn_net, device, norm_status="no_norm"):
    x_raw, y_raw, t_raw, u_raw, v_raw, p_raw, low_bound, up_bound = load_valid_points(filename_raw_data)
    x_pre = x_raw.clone().detach().requires_grad_(True).to(device)
    y_pre = y_raw.clone().detach().requires_grad_(True).to(device)
    t_pre = t_raw.clone().detach().requires_grad_(True).to(device)
    if norm_status == "no_norm":
        u_pre, v_pre, p_pre = pinn_net.predict(x_pre, y_pre, t_pre)
    else:
        u_pre, v_pre, p_pre = pinn_net.predict_inner_norm(x_pre, y_pre, t_pre)
    u_raw_mat = u_raw.numpy()
    v_raw_mat = v_raw.numpy()
    p_raw_mat = p_raw.numpy()
    u_pre_mat = u_pre.cpu().detach().numpy()
    v_pre_mat = v_pre.cpu().detach().numpy()
    p_pre_mat = p_pre.cpu().detach().numpy()
    # 处理数据
    L2_u = (np.linalg.norm(u_pre_mat - u_raw_mat) / np.linalg.norm(u_raw_mat)).reshape(-1, 1)
    L2_v = (np.linalg.norm(v_pre_mat - v_raw_mat) / np.linalg.norm(v_raw_mat)).reshape(-1, 1)
    L2_p = (np.linalg.norm(p_pre_mat - p_raw_mat) / np.linalg.norm(p_raw_mat)).reshape(-1, 1)
    return L2_u, L2_v, L2_p


def inverse_norm(norm_mat, mean_value, std_value):
    real_mat = norm_mat * std_value + mean_value
    return real_mat


def compute_L2_norm_Normalization(filename_raw_data, pinn_net, data_range, device):
    x_raw, y_raw, t_raw, u_raw, v_raw, p_raw, low_bound, up_bound = load_valid_points_Normalization(data_range, filename_raw_data)
    x_pre = x_raw.clone().detach().requires_grad_(True).to(device)
    y_pre = y_raw.clone().detach().requires_grad_(True).to(device)
    t_pre = t_raw.clone().detach().requires_grad_(True).to(device)
    u_pre, v_pre, p_pre = pinn_net.predict(x_pre, y_pre, t_pre)
    u_raw_mat_norm = u_raw.numpy()
    v_raw_mat_norm = v_raw.numpy()
    p_raw_mat_norm = p_raw.numpy()
    u_pre_mat_norm = u_pre.cpu().detach().numpy()
    v_pre_mat_norm = v_pre.cpu().detach().numpy()
    p_pre_mat_norm = p_pre.cpu().detach().numpy()
    # inverse normalization
    # 逆归一化
    u_raw_mat = inverse_norm(u_raw_mat_norm, data_range.mean_u, data_range.std_u)
    v_raw_mat = inverse_norm(v_raw_mat_norm, data_range.mean_v, data_range.std_v)
    p_raw_mat = inverse_norm(p_raw_mat_norm, data_range.mean_p, data_range.std_p)
    u_pre_mat = inverse_norm(u_pre_mat_norm, data_range.mean_u, data_range.std_u)
    v_pre_mat = inverse_norm(v_pre_mat_norm, data_range.mean_v, data_range.std_v)
    p_pre_mat = inverse_norm(p_pre_mat_norm, data_range.mean_p, data_range.std_p)
    L2_u = (np.linalg.norm(u_pre_mat - u_raw_mat) / np.linalg.norm(u_raw_mat)).reshape(-1, 1)
    L2_v = (np.linalg.norm(v_pre_mat - v_raw_mat) / np.linalg.norm(v_raw_mat)).reshape(-1, 1)
    L2_p = (np.linalg.norm(p_pre_mat - p_raw_mat) / np.linalg.norm(p_raw_mat)).reshape(-1, 1)
    return L2_u, L2_v, L2_p


def validation_Baseline(pinn_net, real_path, filename_RL2, device):
    RL2_u, RL2_v, RL2_p = compute_L2_norm(real_path, pinn_net, device, norm_status="no_norm")
    RL2_u_value = RL2_u.reshape(1, 1)
    RL2_v_value = RL2_v.reshape(1, 1)
    RL2_p_value = RL2_p.reshape(1, 1)
    RL2_set = np.concatenate((RL2_u_value, RL2_v_value, RL2_p_value), 1).reshape(1, -1)
    ReL2_save = pd.DataFrame(RL2_set)
    ReL2_save.to_csv(filename_RL2, index=False, header=False, mode='a')
    return RL2_u_value, RL2_v_value, RL2_p_value


def validation_InnerNorm(pinn_net, real_path, filename_RL2, device):
    RL2_u, RL2_v, RL2_p = compute_L2_norm(real_path, pinn_net, device, norm_status="with_norm")
    RL2_u_value = RL2_u.reshape(1, 1)
    RL2_v_value = RL2_v.reshape(1, 1)
    RL2_p_value = RL2_p.reshape(1, 1)
    RL2_set = np.concatenate((RL2_u_value, RL2_v_value, RL2_p_value), 1).reshape(1, -1)
    ReL2_save = pd.DataFrame(RL2_set)
    ReL2_save.to_csv(filename_RL2, index=False, header=False, mode='a')
    return RL2_u_value, RL2_v_value, RL2_p_value


def validation_Normalization(pinn_net, real_path, data_range, filename_RL2, device):
    RL2_u, RL2_v, RL2_p = compute_L2_norm_Normalization(real_path, pinn_net, data_range, device)
    RL2_u_value = RL2_u.reshape(1, 1)
    RL2_v_value = RL2_v.reshape(1, 1)
    RL2_p_value = RL2_p.reshape(1, 1)
    RL2_set = np.concatenate((RL2_u_value, RL2_v_value, RL2_p_value), 1).reshape(1, -1)
    ReL2_save = pd.DataFrame(RL2_set)
    ReL2_save.to_csv(filename_RL2, index=False, header=False, mode='a')
    return RL2_u_value, RL2_v_value, RL2_p_value
