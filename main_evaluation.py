# In[]
import argparse
import torch
from models import OAE
import os
from tqdm import tqdm
import copy
import pandas as pd
import random
from utils import load_training_set, robustness_evaluation_Gaussian, robustness_evaluation_uniform
import os
from main import get_args

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def load_model(args):
     
    filename = f"{args.experiment}_fig{args.figure_num}_layer{args.layer_num}_hidden{args.hidden_size}"
    filename += f"_lr{args.learning_rate}_opt{args.optimizer}_act{args.activation}_sche{args.scheduler}"
    
    if args.noise_std > 0.0 or args.uniform_noise_bound > 0.0:
        filename += f"_ns{args.noise_std}_unb{args.uniform_noise_bound}_mr{args.mask_rate_for_noise_injection}_nd{args.noise_decay}_seed{args.seed}"
    
    if args.noise_std > 0.0 or args.uniform_noise_bound > 0.0:
        filename += f"_ns{args.noise_std}_unb{args.uniform_noise_bound}_mr{args.mask_rate_for_noise_injection}_nd{args.noise_decay}"
    
    model_file = f"{filename}.pth"
    model = torch.load(os.path.join("models", model_file))

    return model
# In[]
args = get_args()
experiments = ['MNIST', 'SVHN', 'CIFAR10', 'TinyImageNet']
# In[]
df_Gaussian_dict = {}
df_uniform_dict = {}
# In[]
for experiment in experiments:
    args.experiment = experiment
    if experiment == 'TinyImageNet':
        args.figure_num = 200
    else:
        args.figure_num = 400
    args.noise_std = 0.0
    args.activation = 'tanh'
    args.learning_rate = 7e-4
    df_Gaussian_dict[f'{experiment}-{args.figure_num}figs'] = {}
    df_uniform_dict[f'{experiment}-{args.figure_num}figs'] = {}

    dataset, input_size, img_size = load_training_set(args)
    imgs = torch.vstack(dataset.imgs)
    for seed in [11, 12, 13, 14, 15, 16]:
        args.seed = seed
        model = load_model(args).cpu().cuda()
        df_Gaussian = robustness_evaluation_Gaussian(model, imgs, max_noise_std=2.0)
        df_uniform = robustness_evaluation_uniform(model, imgs)
        df_Gaussian_dict[f'{experiment}-{args.figure_num}figs']['OAE_seed{}'.format(seed)] = df_Gaussian
        df_uniform_dict[f'{experiment}-{args.figure_num}figs']['OAE_seed{}'.format(seed)] = df_uniform
# In[]
def robustness_evaluation_Gaussian(model, data, max_noise_std=1.0, times=3):
    X0 = data.cuda()
    noise_stds = torch.linspace(0.0, max_noise_std, 11)
    Norms = []
    for noise_std in tqdm(noise_stds, desc='loop 1'):
        norms = []
        for time in tqdm(range(times), desc='loop 2'):
            data_noise = copy.deepcopy(X0) + noise_std*torch.randn_like(X0).cuda()
            X = model.loop(data_noise, step_num=60)
            norm = torch.norm(X.data - X0, dim=1).unsqueeze(dim=1)
            norms.append(norm)
            print(norm.mean().cpu().detach())
        norms = torch.cat(norms, dim=1)
        Norms.append(norms.mean(dim=1).unsqueeze(dim=1))
    Norms = torch.cat(Norms, dim=1)
    df = {}
    df['noise_stds'] = noise_stds.numpy()
    df['norms'] = Norms.cpu().numpy()
    return df

def robustness_evaluation_uniform(model, data, max_mask_rate=1.0, times=1):
    X0 = data.cuda()
    length = X0.shape[1]
    mask_rates = torch.linspace(0.0, max_mask_rate, 11)
    indices = range(0, length)
    Norms = []
    for mask_rate in tqdm(mask_rates, desc='loop 1'):
        norms = []
        mask_num = int(mask_rate*length)
        for time in tqdm(range(times), desc='loop 2'):
            mask_indices = random.sample(indices, mask_num)
            data_mask = copy.deepcopy(X0)
            data_mask[:, mask_indices] = data_mask[:, mask_indices].uniform_(0.0, 1.0)
            X = model.loop(data_mask, step_num=60)
            norm = torch.norm(X.data - X0, dim=1).unsqueeze(dim=1)
            norms.append(norm)
        norms = torch.cat(norms, dim=1)
        Norms.append(norms.mean(dim=1).unsqueeze(dim=1))
    
    Norms = torch.cat(Norms, dim=1)
    df = {}
    df['noise_stds'] = mask_rates.numpy()
    df['norms'] = Norms.cpu().numpy()
    return df
# In[]
datasets = ['MNIST', 'SVHN', 'CIFAR10', 'TinyImageNet']
# model = torch.load('pcnet_models/TinyImageNet_seed1.pt').cuda()

df_Gaussian_NOAE = {}
df_uniform_NOAE = {}
for experiment in datasets:
    args.experiment = experiment
    if experiment == 'TinyImageNet':
        args.figure_num = 200
    else:
        args.figure_num = 400

    args.noise_std = 0.066
    args.mask_rate_for_noise_injection = 0.48
    args.activation = 'tanh'
    args.learning_rate = 5e-4

    dataset, input_size, img_size = load_training_set(args)
    imgs = torch.vstack(dataset.imgs)
    for seed in [11]:
        args.seed = seed
        model = load_model(args).cpu().cuda()
        df_Gaussian = robustness_evaluation_Gaussian(model, imgs, max_noise_std=2.0)
        df_uniform = robustness_evaluation_uniform(model, imgs)
        df_Gaussian_NOAE[f'{experiment}'] = df_Gaussian
        df_uniform_NOAE[f'{experiment}'] = df_uniform

        torch.save(df_Gaussian_NOAE, 'NOAE_Gaussian.pth')
        torch.save(df_uniform_NOAE, 'NOAE_uniform.pth')

# In[]
datasets = ['MNIST', 'SVHN', 'CIFAR10', 'TinyImageNet']
# model = torch.load('pcnet_models/TinyImageNet_seed1.pt').cuda()
df_Gaussian_OAE = {}
df_uniform_OAE = {}
for experiment in datasets:
    args.experiment = experiment
    if experiment == 'TinyImageNet':
        args.figure_num = 200
    else:
        args.figure_num = 400
    args.noise_std = 0.0
    args.activation = 'tanh'
    args.learning_rate = 7e-4

    dataset, input_size, img_size = load_training_set(args)
    imgs = torch.vstack(dataset.imgs)
    for seed in [11]:
        args.seed = seed
        model = load_model(args).cpu().cuda()
        df_Gaussian = robustness_evaluation_Gaussian(model, imgs, max_noise_std=2.0)
        df_uniform = robustness_evaluation_uniform(model, imgs)
        df_Gaussian_OAE[f'{experiment}'] = df_Gaussian
        df_uniform_OAE[f'{experiment}'] = df_uniform

        torch.save(df_Gaussian_OAE, 'OAE_Gaussian.pth')
        torch.save(df_uniform_OAE, 'OAE_uniform.pth')
# In[]
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns

datasets = ['MNIST', 'SVHN', 'CIFAR10', 'TinyImageNet']
df_Gaussian_NOAE = torch.load('NOAE_Gaussian.pth')
PCNet_Gaussian = torch.load('pcnet_gaussian.pth')

# 设置绘图风格
sns.set_theme(style="whitegrid")

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(6, 4.3))

# 定义线条样式和颜色
line_styles = ['-', '--', '-.']
colors = ['blue', 'green', 'red']

# 绘制 NOAE 数据
for idx, exp in enumerate(datasets):
    mean = df_Gaussian_NOAE[exp]['norms'].mean(axis=0)
    std = df_Gaussian_NOAE[exp]['norms'].std(axis=0)
    ax = axs[idx // 2, idx % 2]
    label = 'NOAE' if exp == 'MNIST' else None
    ax.plot(df_Gaussian_NOAE[exp]['noise_stds'], mean, label=label, linestyle=line_styles[0], color=colors[0], linewidth=1.5)
    ax.fill_between(df_Gaussian_NOAE[exp]['noise_stds'], mean - std, mean + std, alpha=0.2, color=colors[0])


# 绘制 OAE 数据
for idx, exp in enumerate(datasets):
    mean = df_Gaussian_OAE[exp]['norms'].mean(axis=0)
    std = df_Gaussian_OAE[exp]['norms'].std(axis=0)
    ax = axs[idx // 2, idx % 2]
    label = 'OAE' if exp == 'MNIST' else None
    ax.plot(df_Gaussian_OAE[exp]['noise_stds'], mean, label=label, linestyle=line_styles[1], color=colors[1], linewidth=1.5)
    ax.fill_between(df_Gaussian_OAE[exp]['noise_stds'], mean - std, mean + std, alpha=0.2, color=colors[1])

# 绘制 GPCN 数据
for idx, exp in enumerate(datasets):
    mean = PCNet_Gaussian[exp]['norms'].mean(axis=0)
    std = PCNet_Gaussian[exp]['norms'].std(axis=0)
    ax = axs[idx // 2, idx % 2]
    label = 'GPCN' if exp == 'MNIST' else None
    ax.plot(df_Gaussian_NOAE[exp]['noise_stds'], mean, label=label, linestyle=line_styles[2], color=colors[2], linewidth=1.5)
    ax.fill_between(df_Gaussian_NOAE[exp]['noise_stds'], mean - 3*std, mean + 3*std, alpha=0.2, color=colors[2])

# 设置图形属性
for idx, exp in enumerate(datasets):
    ax = axs[idx // 2, idx % 2]
    ax.set_ylim(-0.5, 15)
    ax.set_xlim(0.0,)
    ax.set_title(f'{exp}', fontsize=10)
    if idx in [2, 3]:
        ax.set_xlabel('Noise std', fontsize=10)
    if idx in [0, 2]:
        ax.set_ylabel('Discrepancy', fontsize=10)
    ax.legend(fontsize=8)

# 调整布局
plt.tight_layout()
# 保存图形
plt.savefig('figures/GPCN_comparison_Gaussian.jpg', dpi=300)
# 显示图形
plt.show()
# In[]
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns

datasets = ['MNIST', 'SVHN', 'CIFAR10', 'TinyImageNet']
df_uniform_OAE = torch.load('OAE_uniform.pth')
df_uniform_NOAE = torch.load('NOAE_uniform.pth')
df_PCNet_uniform = torch.load('pcnet_uniform.pth')

# 设置绘图风格
sns.set_theme(style="whitegrid")

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(6, 4.3))

# 定义线条样式和颜色
line_styles = ['-', '--', '-.']
colors = ['blue', 'green', 'red']

# 绘制 NOAE 数据
for idx, exp in enumerate(datasets):
    mean = df_uniform_NOAE[exp]['norms'].mean(axis=0)
    std = df_uniform_NOAE[exp]['norms'].std(axis=0)
    ax = axs[idx // 2, idx % 2]
    label = 'NOAE' if exp == 'MNIST' else None
    ax.plot(df_uniform_NOAE[exp]['noise_stds'], mean, label=label, linestyle=line_styles[0], color=colors[0], linewidth=1.5)
    ax.fill_between(df_uniform_NOAE[exp]['noise_stds'], mean - std, mean + std, alpha=0.2, color=colors[0])

# 绘制 OAE 数据
for idx, exp in enumerate(datasets):
    mean = df_uniform_OAE[exp]['norms'].mean(axis=0)
    std = df_uniform_OAE[exp]['norms'].std(axis=0)
    ax = axs[idx // 2, idx % 2]
    label = 'OAE' if exp == 'MNIST' else None
    ax.plot(df_uniform_NOAE[exp]['noise_stds'], mean, label=label, linestyle=line_styles[1], color=colors[1], linewidth=1.5)
    ax.fill_between(df_uniform_OAE[exp]['noise_stds'], mean - std, mean + std, alpha=0.2, color=colors[1])

# 绘制 GPCN 数据
for idx, exp in enumerate(datasets):
    mean = df_PCNet_uniform[exp]['norms'].mean(axis=0)
    std = df_PCNet_uniform[exp]['norms'].std(axis=0)
    ax = axs[idx // 2, idx % 2]
    label = 'GPCN' if exp == 'MNIST' else None
    ax.plot(df_PCNet_uniform[exp]['noise_stds'], mean, label=label, linestyle=line_styles[2], color=colors[2], linewidth=1.5)
    ax.fill_between(df_PCNet_uniform[exp]['noise_stds'], mean - std, mean + std, alpha=0.2, color=colors[2])

# 设置图形属性
for idx, exp in enumerate(datasets):
    ax = axs[idx // 2, idx % 2]
    ax.set_ylim(-0.5, 15)
    ax.set_xlim(0.0,)
    ax.set_title(f'{exp}', fontsize=10)
    if idx in [2, 3]:
        ax.set_xlabel('Mask rate', fontsize=10)
    if idx in [0, 2]:
        ax.set_ylabel('Discrepancy', fontsize=10)
    ax.legend(fontsize=8)

# 调整布局
plt.tight_layout()
# 保存图形
plt.savefig('figures/GPCN_comparison_uniform.jpg', dpi=300)
# 显示图形
plt.show()
# In[]
for experiment in experiments:
    args.experiment = experiment
    if experiment == 'TinyImageNet':
        args.figure_num = 200
    else:
        args.figure_num = 400

    args.noise_std = 0.066
    args.mask_rate_for_noise_injection = 0.48
    args.activation = 'tanh'
    args.learning_rate = 5e-4

    dataset, input_size, img_size = load_training_set(args)
    imgs = torch.vstack(dataset.imgs)
    for seed in [11, 12, 13, 14, 15, 16]:
        args.seed = seed
        model = load_model(args).cpu().cuda()
        df_Gaussian = robustness_evaluation_Gaussian(model, imgs, max_noise_std=2.0)
        df_uniform = robustness_evaluation_uniform(model, imgs)
        df_Gaussian_dict[f'{experiment}-{args.figure_num}figs']['NOAE_seed{}'.format(seed)] = df_Gaussian
        df_uniform_dict[f'{experiment}-{args.figure_num}figs']['NOAE_seed{}'.format(seed)] = df_uniform

# In[]
torch.save(df_Gaussian_dict, 'df_Gaussian_sinificance_test.pth')
torch.save(df_uniform_dict, 'df_uniform_sinificance_test.pth')
# In[]
import matplotlib.pyplot as plt
import seaborn as sns
import torch

df_Gaussian_dict = torch.load('df_Gaussian.pth')
df_uniform_dict = torch.load('df_uniform.pth')

df_Gaussian_hopfield = torch.load('Hopfield-Gaussian.pth')
df_uniform_hopfield = torch.load('Hopfield-uniform.pth')

df_Gaussian_hopfield['MNIST-400figs'] = df_Gaussian_hopfield['MNIST']
df_Gaussian_hopfield['SVHN-400figs'] = df_Gaussian_hopfield['SVHN']
df_Gaussian_hopfield['CIFAR10-400figs'] = df_Gaussian_hopfield['CIFAR10']
df_Gaussian_hopfield['TinyImageNet-200figs'] = df_Gaussian_hopfield['TinyImageNet']

df_uniform_hopfield['MNIST-400figs'] = df_uniform_hopfield['MNIST']
df_uniform_hopfield['SVHN-400figs'] = df_uniform_hopfield['SVHN']
df_uniform_hopfield['CIFAR10-400figs'] = df_uniform_hopfield['CIFAR10']
df_uniform_hopfield['TinyImageNet-200figs'] = df_uniform_hopfield['TinyImageNet']

sns.set_theme(style="whitegrid")

datasets = ['MNIST-400figs', 'SVHN-400figs', 'CIFAR10-400figs', 'TinyImageNet-200figs']
colors = ['#0072B2', '#458BCA', '#87CEEB', '#D55E00', '#FF851B', '#FFBE76', '#8E44AD', '#9B59B6', '#BB8FCE']
markers = ['o', 'o', 'o', 's', 's', 's', '^', '^', '^']

fig, axs = plt.subplots(2, 2, figsize=(8, 6.5))
handles = []
labels = []

for i, dataset in enumerate(datasets):
    ax = axs[i//2, i%2]
    ax.set_title(dataset)
    # 创建一个颜色和标记迭代器
    color_marker_iter = zip(colors, markers)
    
    for key in df_uniform_dict[dataset].keys():
        mean = df_uniform_dict[dataset][key]['correct_rate_means']
        std = df_uniform_dict[dataset][key]['correct_rate_stds']
        if key == 'NOAE-u':
            label = 'NOAE-V2'
        else:
            label = key
        # 获取下一个颜色和标记
        color, marker = next(color_marker_iter)

        line, = ax.plot(df_uniform_dict[dataset][key]['mask_rates'], mean, label=label, color=color, marker=marker)
        ax.fill_between(df_uniform_dict[dataset][key]['mask_rates'], mean - 3*std, mean + 3*std, color=color, alpha=0.2)
        
        # 如果这是第一次遇到这个label，就添加到图例列表中
        if label not in labels:
            handles.append(line)
            labels.append(label)

    for key in df_uniform_hopfield[dataset].keys():
        mean = df_uniform_hopfield[dataset][key]['correct_rate_means']
        std = df_uniform_hopfield[dataset][key]['correct_rate_stds']
        # 再次获取下一个颜色和标记
        color, marker = next(color_marker_iter)
        line, = ax.plot(df_uniform_hopfield[dataset][key]['mask_rates'], mean, label=key, color=color, marker=marker)
        ax.fill_between(df_uniform_hopfield[dataset][key]['mask_rates'], mean - 3*std, mean + 3*std, color=color, alpha=0.2)
        
        # 如果这是第一次遇到这个label，就添加到图例列表中
        if key not in labels:
            handles.append(line)
            labels.append(key)
    
    # 检查图例是否存在，如果存在则删除
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

axs[0,0].set(xlabel=None, ylabel='Correct rate')
axs[0,1].set(xlabel=None, ylabel=None)
axs[1,0].set(xlabel='Mask rate', ylabel='Correct rate')
axs[1,1].set(xlabel='Mask rate', ylabel=None)
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02)) 
plt.tight_layout(rect=[0, 0.02, 1, 0.90])
plt.savefig('uniform_noise.jpg', dpi=600)
# In[]
fig, axs = plt.subplots(2, 2, figsize=(8, 6.5))
handles = []
labels = []

for i, dataset in enumerate(datasets):
    ax = axs[i//2, i%2]
    ax.set_title(dataset)
    # 创建一个颜色和标记迭代器
    color_marker_iter = zip(colors, markers)
    
    for key in df_Gaussian_dict[dataset].keys():
        mean = df_Gaussian_dict[dataset][key]['correct_rate_means']
        std = df_Gaussian_dict[dataset][key]['correct_rate_stds']
        if key == 'NOAE-u':
            label = 'NOAE-V2'
        else:
            label = key
        # 获取下一个颜色和标记
        color, marker = next(color_marker_iter)

        line, = ax.plot(df_Gaussian_dict[dataset][key]['noise_stds'], mean, label=label, color=color, marker=marker)
        ax.fill_between(df_Gaussian_dict[dataset][key]['noise_stds'], mean - 3*std, mean + 3*std, color=color, alpha=0.2)
        
        # 如果这是第一次遇到这个label，就添加到图例列表中
        if label not in labels:
            handles.append(line)
            labels.append(label)

    for key in df_Gaussian_hopfield[dataset].keys():
        mean = df_Gaussian_hopfield[dataset][key]['correct_rate_means']
        std = df_Gaussian_hopfield[dataset][key]['correct_rate_stds']
        # 再次获取下一个颜色和标记
        color, marker = next(color_marker_iter)
        line, = ax.plot(df_Gaussian_hopfield[dataset][key]['noise_stds'], mean, label=key, color=color, marker=marker)
        ax.fill_between(df_Gaussian_hopfield[dataset][key]['noise_stds'], mean - 3*std, mean + 3*std, color=color, alpha=0.2)
        
        # 如果这是第一次遇到这个label，就添加到图例列表中
        if key not in labels:
            handles.append(line)
            labels.append(key)
    
    # 检查图例是否存在，如果存在则删除
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

axs[0,0].set(xlabel=None, ylabel='Correct rate')
axs[0,1].set(xlabel=None, ylabel=None)
axs[1,0].set(xlabel='Noise std', ylabel='Correct rate')
axs[1,1].set(xlabel='Noise std', ylabel=None)
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02)) 
plt.tight_layout(rect=[0, 0.02, 1, 0.90])
plt.savefig('Gaussian_noise.jpg', dpi=600)

# In[]
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# 加载数据
df_Gaussian_dict = torch.load('df_Gaussian.pth')
df_uniform_dict = torch.load('df_uniform.pth')
df_Gaussian_hopfield = torch.load('Hopfield-Gaussian.pth')
df_uniform_hopfield = torch.load('Hopfield-uniform.pth')

# 数据预处理
for df in [df_Gaussian_hopfield, df_uniform_hopfield]:
    df['MNIST-400figs'] = df['MNIST']
    df['SVHN-400figs'] = df['SVHN']
    df['CIFAR10-400figs'] = df['CIFAR10']
    df['TinyImageNet-200figs'] = df['TinyImageNet']

sns.set_theme(style="whitegrid")

# 设置绘图参数
datasets = ['MNIST-400figs', 'SVHN-400figs', 'CIFAR10-400figs', 'TinyImageNet-200figs']
colors = ['#0072B2', '#458BCA', '#87CEEB', '#D55E00', '#FF851B', '#FFBE76', '#8E44AD', '#9B59B6', '#BB8FCE']
markers = ['o', 'o', 'o', 's', 's', 's', '^', '^', '^']

# 创建图表和子图
fig, axs = plt.subplots(2,4, figsize=(14, 7), sharex='col', sharey='row')
handles = []
labels = []

# 绘制高斯噪声数据
for i, dataset in enumerate(datasets):
    ax = axs[0, i]  # 上半部分是高斯噪声
    ax.set_xlim(0, 2.0)
    ax.set_title(f'{dataset} (Gaussian)')
    color_marker_iter = zip(colors, markers)
    
    for key in df_Gaussian_dict[dataset].keys():
        mean = df_Gaussian_dict[dataset][key]['correct_rate_means']
        std = df_Gaussian_dict[dataset][key]['correct_rate_stds']
        label = 'NOAE-V2' if key == 'NOAE-u' else key
        color, marker = next(color_marker_iter)
        line, = ax.plot(df_Gaussian_dict[dataset][key]['noise_stds'], mean, label=label, color=color, marker=marker)
        ax.fill_between(df_Gaussian_dict[dataset][key]['noise_stds'], mean - 3*std, mean + 3*std, color=color, alpha=0.2)
        if label not in labels:
            handles.append(line)
            labels.append(label)

    for key in df_Gaussian_hopfield[dataset].keys():
        mean = df_Gaussian_hopfield[dataset][key]['correct_rate_means']
        std = df_Gaussian_hopfield[dataset][key]['correct_rate_stds']
        color, marker = next(color_marker_iter)
        line, = ax.plot(df_Gaussian_hopfield[dataset][key]['noise_stds'], mean, label=key, color=color, marker=marker)
        ax.fill_between(df_Gaussian_hopfield[dataset][key]['noise_stds'], mean - 3*std, mean + 3*std, color=color, alpha=0.2)
        if key not in labels:
            handles.append(line)
            labels.append(key)

# 绘制均匀噪声数据
for i, dataset in enumerate(datasets):
    ax = axs[1, i]  # 下半部分是均匀噪声
    ax.set_xlim(0, 1.0)
    ax.set_title(f'{dataset} (Uniform)')
    color_marker_iter = zip(colors, markers)
    
    for key in df_uniform_dict[dataset].keys():
        mean = df_uniform_dict[dataset][key]['correct_rate_means']
        std = df_uniform_dict[dataset][key]['correct_rate_stds']
        label = 'NOAE-V2' if key == 'NOAE-u' else key
        color, marker = next(color_marker_iter)
        line, = ax.plot(df_uniform_dict[dataset][key]['mask_rates'], mean, label=label, color=color, marker=marker)
        ax.fill_between(df_uniform_dict[dataset][key]['mask_rates'], mean - 3*std, mean + 3*std, color=color, alpha=0.2)
        if label not in labels:
            handles.append(line)
            labels.append(label)

    for key in df_uniform_hopfield[dataset].keys():
        mean = df_uniform_hopfield[dataset][key]['correct_rate_means']
        std = df_uniform_hopfield[dataset][key]['correct_rate_stds']
        color, marker = next(color_marker_iter)
        line, = ax.plot(df_uniform_hopfield[dataset][key]['mask_rates'], mean, label=key, color=color, marker=marker)
        ax.fill_between(df_uniform_hopfield[dataset][key]['mask_rates'], mean - 3*std, mean + 3*std, color=color, alpha=0.2)
        if key not in labels:
            handles.append(line)
            labels.append(key)

# 设置轴标签
for ax in axs[1, :]:  # 只在下半部分设置xlabel
    ax.set_xlabel('Mask rate')
    ax.set_xlim(0, 1.0)
for ax in axs[0, :]:  # 只在下半部分设置xlabel
    ax.set_xlabel('Noise std')
    ax.set_xlim(0, 2.0)
for ax in axs[:, 0]:  # 只在最左边一列设置ylabel
    ax.set_ylabel('Correct rate')


# 添加全局图例
fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))

# 调整布局
plt.tight_layout(rect=[0, 0.02, 1, 0.90])

# 保存图片
plt.savefig('combined_noise_plots.jpg', dpi=400)

# 显示图表
plt.show()
# In[]
datasets = ['MNIST', 'SVHN', 'CIFAR10', 'TinyImageNet']
for i, dataset in enumerate(datasets):
    ax = axs[i//2, i%2]
    ax.set_title(dataset)

colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#C10020', '#56B4E9']
markers = ['o', 's', '^', 'd', '*', 'v']

handles = []
labels = []

for i, dataset in enumerate(datasets):
    ax = axs[i//2, i%2]
    
    line1, = ax.plot(df_uniform_dict[dataset]['OAE']['mask_rates'], df_uniform_dict[dataset]['OAE']['correct_rate_means'], label='OAE', color=colors[0], marker=markers[0])
    line2, = ax.plot(df_uniform_dict[dataset]['NOAE']['mask_rates'], df_uniform_dict[dataset]['NOAE']['correct_rate_means'], label='NOAE', color=colors[1], marker=markers[1])
    line3, = ax.plot(df_uniform_hopfield[dataset]['Euclidean']['mask_rates'], df_uniform_hopfield[dataset]['Euclidean']['correct_rate_means'], label='Hopfield-Euclidean', color=colors[2], marker=markers[2])
    line4, = ax.plot(df_uniform_hopfield[dataset]['Manhatten']['mask_rates'], df_uniform_hopfield[dataset]['Manhatten']['correct_rate_means'], label='Hopfield-Manhatten', color=colors[3], marker=markers[3])
    line5, = ax.plot(df_uniform_hopfield[dataset]['dot_product']['mask_rates'], df_uniform_hopfield[dataset]['dot_product']['correct_rate_means'], label='Hopfield-dot_product', color=colors[4], marker=markers[4])

    line6, = ax.plot(df_uniform_dict[dataset]['NOAE-u']['mask_rates'], df_uniform_dict[dataset]['NOAE-u']['correct_rate_means'], label='NOAE-V2', color=colors[5], marker=markers[5])
    
    ax.fill_between(df_uniform_dict[dataset]['OAE']['mask_rates'], df_uniform_dict[dataset]['OAE']['correct_rate_means'] - 3*df_uniform_dict[dataset]['OAE']['correct_rate_stds'], df_uniform_dict[dataset]['OAE']['correct_rate_means'] + 3*df_uniform_dict[dataset]['OAE']['correct_rate_stds'], alpha=0.2, color=colors[0])
    ax.fill_between(df_uniform_dict[dataset]['NOAE']['mask_rates'], df_uniform_dict[dataset]['NOAE']['correct_rate_means'] - 3*df_uniform_dict[dataset]['NOAE']['correct_rate_stds'], df_uniform_dict[dataset]['NOAE']['correct_rate_means'] + 3*df_uniform_dict[dataset]['NOAE']['correct_rate_stds'], alpha=0.2, color=colors[1])
    ax.fill_between(df_uniform_hopfield[dataset]['Euclidean']['mask_rates'], df_uniform_hopfield[dataset]['Euclidean']['correct_rate_means'] - 3*df_uniform_hopfield[dataset]['Euclidean']['correct_rate_stds'], df_uniform_hopfield[dataset]['Euclidean']['correct_rate_means'] + 3*df_uniform_hopfield[dataset]['Euclidean']['correct_rate_stds'], alpha=0.2, color=colors[2])
    ax.fill_between(df_uniform_hopfield[dataset]['Manhatten']['mask_rates'], df_uniform_hopfield[dataset]['Manhatten']['correct_rate_means'] - 3*df_uniform_hopfield[dataset]['Manhatten']['correct_rate_stds'], df_uniform_hopfield[dataset]['Manhatten']['correct_rate_means'] + 3*df_uniform_hopfield[dataset]['Manhatten']['correct_rate_stds'], alpha=0.2, color=colors[3])
    ax.fill_between(df_uniform_hopfield[dataset]['dot_product']['mask_rates'], df_uniform_hopfield[dataset]['dot_product']['correct_rate_means'] - 3*df_uniform_hopfield[dataset]['dot_product']['correct_rate_stds'], df_uniform_hopfield[dataset]['dot_product']['correct_rate_means'] + 3*df_uniform_hopfield[dataset]['dot_product']['correct_rate_stds'], alpha=0.2, color=colors[4])
    ax.fill_between(df_uniform_dict[dataset]['NOAE-u']['mask_rates'], df_uniform_dict[dataset]['NOAE-u']['correct_rate_means'] - 3*df_uniform_dict[dataset]['NOAE-u']['correct_rate_stds'], df_uniform_dict[dataset]['NOAE-u']['correct_rate_means'] + 3*df_uniform_dict[dataset]['NOAE-u']['correct_rate_stds'], alpha=0.2, color=colors[5])

datasets = ['MNIST', 'SVHN', 'CIFAR10', 'TinyImageNet']
for i, dataset in enumerate(datasets):
    ax = axs[i//2, i%2]
    ax.set_title(dataset)
    

handles.append(line1)
handles.append(line2)
handles.append(line3)
handles.append(line4)
handles.append(line5)
handles.append(line6)
labels.append('OAE')
labels.append('NOAE')
labels.append('Hopfield-Euclidean')
labels.append('Hopfield-Manhatten')
labels.append('Hopfield-dot_product')
labels.append('NOAE-V2')

axs[0,0].set(xlabel=None, ylabel='Correct rate')
axs[0,1].set(xlabel=None, ylabel=None)
axs[1,0].set(xlabel='Mask rate', ylabel='Correct rate')
axs[1,1].set(xlabel='Mask rate', ylabel=None)

# 添加统一的图例
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98)) 
plt.tight_layout(rect=[0, 0.02, 1, 1.05])
plt.savefig('uniform_noise.jpg', dpi=400)
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
# plt.show()
# %%
# 设置seaborn风格
sns.set(style="whitegrid")

fig, axs = plt.subplots(2, 2, figsize=(7, 5.5))

datasets = ['MNIST-400figs', 'SVHN-400figs', 'CIFAR10-400figs', 'TinyImageNet-200figs']

colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#C10020', '#56B4E9']
markers = ['o', 's', '^', 'd', '*', 'v']

handles = []
labels = []

for i, dataset in enumerate(datasets):
    ax = axs[i//2, i%2]
    line1, = ax.plot(df_Gaussian_dict[dataset]['OAE']['noise_stds'], df_Gaussian_dict[dataset]['OAE']['correct_rate_means'], label='OAE mean', color=colors[0], marker=markers[0])
    line2, = ax.plot(df_Gaussian_dict[dataset]['NOAE']['noise_stds'], df_Gaussian_dict[dataset]['NOAE']['correct_rate_means'], label='NOAE mean', color=colors[1], marker=markers[1])
    line3, = ax.plot(df_Gaussian_hopfield[dataset]['Euclidean']['noise_stds'], df_Gaussian_hopfield[dataset]['Euclidean']['correct_rate_means'], label='Hopfield-Euclidean mean', color=colors[2], marker=markers[2])
    line4, = ax.plot(df_Gaussian_hopfield[dataset]['Manhatten']['noise_stds'], df_Gaussian_hopfield[dataset]['Manhatten']['correct_rate_means'], label='Hopfield-Manhatten mean', color=colors[3], marker=markers[3])
    line5, = ax.plot(df_Gaussian_hopfield[dataset]['dot_product']['noise_stds'], df_Gaussian_hopfield[dataset]['dot_product']['correct_rate_means'], label='Hopfield-dot_product mean', color=colors[4], marker=markers[4])
    line2, = ax.plot(df_Gaussian_dict[dataset]['NOAE-u']['noise_stds'], df_Gaussian_dict[dataset]['NOAE-u']['correct_rate_means'], label='NOAE mean', color=colors[5], marker=markers[5])
    
    
    ax.fill_between(df_Gaussian_dict[dataset]['OAE']['noise_stds'], df_Gaussian_dict[dataset]['OAE']['correct_rate_means'] - 3*df_Gaussian_dict[dataset]['OAE']['correct_rate_stds'], df_Gaussian_dict[dataset]['OAE']['correct_rate_means'] + 3*df_Gaussian_dict[dataset]['OAE']['correct_rate_stds'], alpha=0.2, color=colors[0])
    ax.fill_between(df_Gaussian_dict[dataset]['NOAE']['noise_stds'], df_Gaussian_dict[dataset]['NOAE']['correct_rate_means'] - 3*df_Gaussian_dict[dataset]['NOAE']['correct_rate_stds'], df_Gaussian_dict[dataset]['NOAE']['correct_rate_means'] + 3*df_Gaussian_dict[dataset]['NOAE']['correct_rate_stds'], alpha=0.2, color=colors[1])
    ax.fill_between(df_Gaussian_hopfield[dataset]['Euclidean']['noise_stds'], df_Gaussian_hopfield[dataset]['Euclidean']['correct_rate_means'] - 3*df_Gaussian_hopfield[dataset]['Euclidean']['correct_rate_stds'], df_Gaussian_hopfield[dataset]['Euclidean']['correct_rate_means'] + 3*df_uniform_hopfield[dataset]['Euclidean']['correct_rate_stds'], alpha=0.2, color=colors[2])
    ax.fill_between(df_Gaussian_hopfield[dataset]['Manhatten']['noise_stds'], df_Gaussian_hopfield[dataset]['Manhatten']['correct_rate_means'] - 3*df_Gaussian_hopfield[dataset]['Manhatten']['correct_rate_stds'], df_Gaussian_hopfield[dataset]['Manhatten']['correct_rate_means'] + 3*df_uniform_hopfield[dataset]['Manhatten']['correct_rate_stds'], alpha=0.2, color=colors[3])
    ax.fill_between(df_Gaussian_hopfield[dataset]['dot_product']['noise_stds'], df_Gaussian_hopfield[dataset]['dot_product']['correct_rate_means'] - 3*df_Gaussian_hopfield[dataset]['dot_product']['correct_rate_stds'], df_Gaussian_hopfield[dataset]['dot_product']['correct_rate_means'] + 3*df_uniform_hopfield[dataset]['dot_product']['correct_rate_stds'], alpha=0.2, color=colors[4])
    ax.fill_between(df_Gaussian_dict[dataset]['NOAE-u']['noise_stds'], df_Gaussian_dict[dataset]['NOAE-u']['correct_rate_means'] - 3*df_Gaussian_dict[dataset]['NOAE-u']['correct_rate_stds'], df_Gaussian_dict[dataset]['NOAE-u']['correct_rate_means'] + 3*df_Gaussian_dict[dataset]['NOAE-u']['correct_rate_stds'], alpha=0.2, color=colors[5])

datasets = ['MNIST', 'SVHN', 'CIFAR10', 'TinyImageNet']
for i, dataset in enumerate(datasets):
    ax = axs[i//2, i%2]
    ax.set_title(dataset)
    
handles.append(line1)
handles.append(line2)
handles.append(line3)
handles.append(line4)
handles.append(line5)
handles.append(line6)
labels.append('OAE')
labels.append('NOAE')
labels.append('Hopfield-Euclidean')
labels.append('Hopfield-Manhatten')
labels.append('Hopfield-dot_product')
labels.append('NOAE-V2')

axs[0,0].set(xlabel=None, ylabel='Correct rate')
axs[0,1].set(xlabel=None, ylabel=None)
axs[1,0].set(xlabel='Noise std', ylabel='Correct rate')
axs[1,1].set(xlabel='Noise std', ylabel=None)

# 添加统一的图例
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98)) 
plt.tight_layout(rect=[0, 0.02, 1, 0.90])
plt.savefig('Gaussian_noise.jpg', dpi=400)