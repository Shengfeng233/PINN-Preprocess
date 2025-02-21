import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.edgecolor'] = 'black'
title_size = 16
label_size = 14
legend_size = 11
title_dict = {'family': 'Times New Roman', 'weight': 'bold', 'size': title_size}
label_dict = {'family': 'Times New Roman', 'weight': 'bold', 'size': label_size}
legend_dict = {'family': 'Times New Roman', 'weight': 'bold', 'size': legend_size}


def plot_bar_RL2_uvp(values, deviations, legend_set, case):
    categories = legend_set
    x = np.arange(len(categories))
    bar_width = 0.15
    bar_offset = 0.15
    color_1 = '#a1c9f4'
    color_2 = '#ffb482'
    color_3 = '#8de5a1'
    edge_color = 'white'
    std_color = 'grey'
    edge_width = 0.5
    capsize = 2.5
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    # sub fig 1
    values_u_1 = values[0, :, 0]
    values_u_2 = values[1, :, 0]
    values_u_3 = values[2, :, 0]
    std_u_1 = deviations[0, :, 0]
    std_u_2 = deviations[1, :, 0]
    std_u_3 = deviations[2, :, 0]
    axs[0].bar(x - bar_offset, values_u_1, bar_width, color=color_1, edgecolor=edge_color, linewidth=edge_width, yerr=std_u_1, capsize=capsize, ecolor=std_color)
    axs[0].bar(x, values_u_2, bar_width, color=color_2, edgecolor=edge_color, linewidth=edge_width, yerr=std_u_2, capsize=capsize, ecolor=std_color)
    axs[0].bar(x + bar_offset, values_u_3, bar_width, color=color_3, edgecolor=edge_color, linewidth=edge_width, yerr=std_u_3, capsize=capsize, ecolor=std_color)
    axs[0].set_title('Relative $\mathrm{L_2}$ norm - u', fontdict=title_dict)
    axs[0].set_ylabel('Relative $\mathrm{L_2}$ norm', fontdict=label_dict)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(categories, fontdict=label_dict)
    for label in axs[0].get_yticklabels():
        label.set_weight('bold')
        label.set_size(label_size)
    axs[0].legend(['Baseline', 'InnerNorm', 'Normalization'], loc='lower left', prop=legend_dict)

    # sub fig 2
    values_v_1 = values[0, :, 1]
    values_v_2 = values[1, :, 1]
    values_v_3 = values[2, :, 1]
    std_v_1 = deviations[0, :, 1]
    std_v_2 = deviations[1, :, 1]
    std_v_3 = deviations[2, :, 1]
    axs[1].bar(x - bar_offset, values_v_1, bar_width, color=color_1, edgecolor=edge_color, linewidth=edge_width, yerr=std_v_1, capsize=capsize, ecolor=std_color)
    axs[1].bar(x, values_v_2, bar_width, color=color_2, edgecolor=edge_color, linewidth=edge_width, yerr=std_v_2, capsize=capsize, ecolor=std_color)
    axs[1].bar(x + bar_offset, values_v_3, bar_width, color=color_3, edgecolor=edge_color, linewidth=edge_width, yerr=std_v_3, capsize=capsize, ecolor=std_color)
    axs[1].set_title('Relative $\mathrm{L_2}$ norm - v', fontdict=title_dict)
    axs[1].set_ylabel('Relative $\mathrm{L_2}$ norm', fontdict=label_dict)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(categories, fontdict=label_dict)
    for label in axs[1].get_yticklabels():
        label.set_weight('bold')
        label.set_size(label_size)
    axs[1].legend(['Baseline', 'InnerNorm', 'Normalization'], loc='lower left', prop=legend_dict)
    # sub fig 3
    values_p_1 = values[0, :, 2]
    values_p_2 = values[1, :, 2]
    values_p_3 = values[2, :, 2]
    std_p_1 = deviations[0, :, 2]
    std_p_2 = deviations[1, :, 2]
    std_p_3 = deviations[2, :, 2]
    axs[2].bar(x - bar_offset, values_p_1, bar_width, color=color_1, edgecolor=edge_color, linewidth=edge_width, yerr=std_p_1, capsize=capsize, ecolor=std_color)
    axs[2].bar(x, values_p_2, bar_width, color=color_2, edgecolor=edge_color, linewidth=edge_width, yerr=std_p_2, capsize=capsize, ecolor=std_color)
    axs[2].bar(x + bar_offset, values_p_3, bar_width, color=color_3, edgecolor=edge_color, linewidth=edge_width, yerr=std_p_3, capsize=capsize, ecolor=std_color)
    axs[2].set_title('Relative $\mathrm{L_2}$ norm - p', fontdict=title_dict)
    axs[2].set_ylabel('Relative $\mathrm{L_2}$ norm', fontdict=label_dict)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(categories, fontdict=label_dict)
    for label in axs[2].get_yticklabels():
        label.set_weight('bold')
        label.set_size(label_size)
    axs[2].legend(['Baseline', 'InnerNorm', 'Normalization'], loc='lower left', prop=legend_dict)

    # show figure
    # 显示图表
    plt.tight_layout()
    plt.savefig('Fig_8_batch_size_compare_ave' + case + '.png', bbox_inches='tight', dpi=100, pad_inches=0.05)
    plt.show()
    return


def plot_bar_RL2_uvp_multi_case(preprocess_set, dataset, batch_size_set, hidden_layers, layer_neurons, scheduler, rep_set):
    for case in dataset:
        value_set = np.empty([len(preprocess_set), len(batch_size_set), len(rep_set), 3])
        for index_i, preprocess in enumerate(preprocess_set):
            for index_j, batch_size in enumerate(batch_size_set):
                for index_k, rep_name in enumerate(rep_set):
                    write_path = './write_sweep/batch_size/{}_Case_{}_Preprocess_{}_Layer_{}_Neuron_{}_BatchSize_{}_Scheduler_{}' \
                        .format(rep_name, case, preprocess, hidden_layers, layer_neurons, batch_size, scheduler)
                    evaluate_path = write_path + '/RL2.csv'
                    RL2_uvp = pd.read_csv(evaluate_path, header=0, index_col=None).values.reshape(-1, 3)
                    RL2_uvp_values = RL2_uvp[-2, :].reshape(-1, 3)
                    value_set[index_i, index_j, index_k, :] = RL2_uvp_values

        legend_set = ['Batch: 2048', 'Batch: 8192', 'Batch: 32768']
        mean_values = np.mean(value_set, axis=2)
        std_values = np.std(value_set, axis=2)
        plot_bar_RL2_uvp(mean_values, std_values, legend_set, case)
    return


preprocess_set = ['Baseline', 'InnerNorm', 'Normalization']
dataset = ['Re48000000', 'Re3900', 'Re10000']
batch_size_set = [2048, 8192, 32768]
hidden_layers = 10
layer_neurons = 32
scheduler = 'exp'
rep_set = ['Rep1', 'Rep2', 'Rep3']

if __name__ == '__main__':
    plot_bar_RL2_uvp_multi_case(preprocess_set, dataset, batch_size_set, hidden_layers, layer_neurons, scheduler, rep_set)

