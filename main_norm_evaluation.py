# In[]
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
from tqdm import tqdm
from main_norm import *
import pandas as pd
import seaborn as sns

torch.cuda.set_device('cuda:5')

# 上面提供的函数（Gaussian_noise_distortion, robustness_evaluation_Gaussian, uniform_noise_distortion, robustness_evaluation_uniform, top_mask_distortion, robustness_evaluation_top_mask, center_random_mask, robustness_evaluation_center_random_mask）应在此处定义
def transformations(imgs, img_size, scale_coefficient=1.0, index=None):
    if index == None:
        imgs = imgs.reshape(-1, *img_size)
    else:
        imgs = imgs.reshape(-1, *img_size)[index].unsqueeze(dim=0)

    # 获取原始图像
    original_img = imgs.squeeze()
    # 应用灰色噪声
    gray_img = gray_distortion(imgs, 0.3).squeeze()
    blue_img = blue_distortion(imgs, 2.0).squeeze()

    # 应用顶部区域掩码
    top_black_img = top_mask_distortion(imgs, img_size, 0.4, 'black', scale_coefficient).squeeze()
    diag_gray_img = diagonal_mask_distortion(imgs, img_size, 7, 'gray', scale_coefficient).squeeze()

    # 应用中心随机掩码
    center_red_img = center_random_mask(imgs, img_size, 0.3, 'red', scale_coefficient).squeeze()
    center_black_img = center_random_mask(imgs, img_size, 0.3, 'black', scale_coefficient).squeeze()
    center_noise_img = center_random_mask(imgs, img_size, 0.5, 'noise', scale_coefficient).squeeze()

    return (original_img, gray_img, blue_img, top_black_img, diag_gray_img, center_red_img, center_black_img, center_noise_img)

# In[]
referenced_file_name = "OAE_CIFAR10_Fig50_Norm_Bound4.0_H512_L7_E200000_LR0.001_OptAdam_Acttanh_SchCosine_Noise0.0_Seed0.pt"
args = parse_filename_to_args(referenced_file_name)
args.figure_num = 200
args.activation = "tanh"
args.experiment = 'TinyImageNet'
args.learning_rate = 7e-4
# 初始化数据列表
df_original = []
df_gray = []
df_blue = []
df_top_black = []
df_diag_gray = []
df_center_red = []
df_center_black = []
df_center_noise = []
# 定义一个通用函数来处理每个列表的填充
def process_dataframes(injected_noise_std, norm_bound, figure_num, args):
    args.noise_std = injected_noise_std
    args.norm_bound = norm_bound
    args.figure_num = figure_num

    # 加载数据集和模型
    trainset, input_size, img_size, scale_coefficient, img_indices = load_training_set(args)
    imgs = trainset.data.cuda()
    model_filename = generate_model_filename(args)
    model_path = os.path.join("models", model_filename)
    model = torch.load(model_path).cuda()

    # 获取所有变换后的图像
    (original_img, gray_img, blue_img, top_black_img, diag_gray_img, center_red_img, center_black_img, center_noise_img) = \
        transformations(imgs, img_size, scale_coefficient, index=None)

    # 定义一个处理单个图像变换的函数
    def process_image(image, dataframe, name):
        queried_imgs = model.loop(image.cuda().reshape(imgs.size(0), -1), step_num=60)
        norm_discrepancy = torch.norm(queried_imgs - imgs, dim=1).detach().cpu().numpy()
        for norm in norm_discrepancy:
            dataframe.append({
                'norm_bound': norm_bound,
                'norm': norm,
                'injected_noise_std': injected_noise_std,
                'figure_num':figure_num
            })
    # 处理每个图像变换
    process_image(original_img, df_original, "original")
    process_image(gray_img, df_gray, 'gray')
    process_image(blue_img, df_blue, 'blue')
    process_image(top_black_img, df_top_black, 'top_black')
    process_image(diag_gray_img, df_diag_gray, 'diag_gray')
    process_image(center_red_img, df_center_red, 'center_red')
    process_image(center_black_img, df_center_black, 'center_black')
    process_image(center_noise_img, df_center_noise, 'center_uniform')
# 循环处理每种噪声标准差和范数边界组合
for figure_num in [50, 100, 200, 400, 800, 1600]:
    for injected_noise_std in [0.00, 0.05, 0.10, 0.20]:
        for norm_bound in [4.0, 8.0, 16.0, 32.0]:
            process_dataframes(injected_noise_std, norm_bound, figure_num, args)
# 转换数据列表为DataFrame
df_original = pd.DataFrame(df_original)
df_gray = pd.DataFrame(df_gray)
df_blue = pd.DataFrame(df_blue)
df_top_black = pd.DataFrame(df_top_black)
df_diag_gray = pd.DataFrame(df_diag_gray)
df_center_red = pd.DataFrame(df_center_red)
df_center_black = pd.DataFrame(df_center_black)
df_center_noise = pd.DataFrame(df_center_noise)

# 定义保存路径
save_path = f"dataframes"
os.makedirs(save_path, exist_ok=True)

# 保存每个 DataFrame
df_original.to_pickle(os.path.join(save_path, "df_original.pkl"))
df_gray.to_pickle(os.path.join(save_path, "df_gray.pkl"))
df_blue.to_pickle(os.path.join(save_path, "df_blue.pkl"))
df_top_black.to_pickle(os.path.join(save_path, "df_top_black.pkl"))
df_diag_gray.to_pickle(os.path.join(save_path, "df_diag_gray.pkl"))
df_center_red.to_pickle(os.path.join(save_path, "df_center_red.pkl"))
df_center_black.to_pickle(os.path.join(save_path, "df_center_black.pkl"))
df_center_noise.to_pickle(os.path.join(save_path, "df_center_noise.pkl"))   
print("DataFrames have been saved.")
# In[]
# 定义加载路径
load_path = f"dataframes"

# 加载每个 DataFrame
df_original = pd.read_pickle(os.path.join(load_path, "df_original.pkl"))
df_gray = pd.read_pickle(os.path.join(load_path, "df_gray.pkl"))
df_blue = pd.read_pickle(os.path.join(load_path, "df_blue.pkl"))
df_top_black = pd.read_pickle(os.path.join(load_path, "df_top_black.pkl"))
df_diag_gray = pd.read_pickle(os.path.join(load_path, "df_diag_gray.pkl"))
df_center_red = pd.read_pickle(os.path.join(load_path, "df_center_red.pkl"))
df_center_black = pd.read_pickle(os.path.join(load_path, "df_center_black.pkl"))
df_center_noise = pd.read_pickle(os.path.join(load_path, "df_center_noise.pkl"))

print("DataFrames have been loaded.")
# In[]
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
import matplotlib.pyplot as plt

# 设定 p 值阈值
p_value_threshold = 0.05  # 您可以根据需要调整这个值

# 不同的 figure_num 值
figure_nums = [200, 400, 800, 1600]
norm_bounds = [4.0, 8.0, 16.0, 32.0]
transformations = ['original image', 'Low brightness', 'High blue', 'Top black mask', 'Diagonal gray mask', 'Center orange mask', 'Center black mask', 'Center uniform mask']

# 定义自定义颜色映射
cmap = plt.get_cmap('coolwarm')

# 创建一个全局的颜色条
def create_colorbar(fig, im):
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Null hypothesis is rejected')

# 绘制热力图函数
def plot_heatmap(noise_std, dfs, filename):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(13, 3))
    axes = axes.ravel()  # 将1D数组转换为1D，方便循环

    for idx, fixed_figure_num in enumerate(figure_nums):
        comparison_results = {transformation: {} for transformation in transformations}
        
        for norm_bound in norm_bounds:
            for transformation, df in zip(transformations, dfs):
                data_0_0 = df[(df['figure_num'] == fixed_figure_num) & 
                              (df['norm_bound'] == norm_bound) & 
                              (df['injected_noise_std'] == 0.0)]['norm'].values
                data_0_x = df[(df['figure_num'] == fixed_figure_num) & 
                              (df['norm_bound'] == norm_bound) & 
                              (df['injected_noise_std'] == noise_std)]['norm'].values
                
                epsilon = 1e-10  # 防止取对数时出现负数或零
                log_data_0_0 = np.log(data_0_0 + epsilon)
                log_data_0_x = np.log(data_0_x + epsilon)
                
                if len(log_data_0_0) > 0 and len(log_data_0_x) > 0:
                    stat, p_value = mannwhitneyu(log_data_0_x, log_data_0_0, alternative='less')
                    
                    if p_value < p_value_threshold:
                        result = 1  # injected_noise_std 显著小于 0.0
                    else:
                        result = 0  # 无显著差异或 injected_noise_std=0.0 更好
                else:
                    result = 0
                    
                comparison_results[transformation][norm_bound] = result

        comparison_results_df = pd.DataFrame(comparison_results).T

        im = axes[idx].imshow(
            comparison_results_df.values, 
            cmap=cmap, 
            aspect='auto',
            vmin=0,
            vmax=1
        )

        for i in range(len(transformations)):
            for j in range(len(norm_bounds)):
                text = axes[idx].text(j, i, int(comparison_results_df.iloc[i, j]),
                                     ha="center", va="center", color="white")

        axes[idx].set_title(f'Figure Num {fixed_figure_num}', pad=5)
        axes[idx].set_xlabel('Norm Bound', labelpad=5)

        if idx == 0:
            axes[idx].set_ylabel('Image Distortion')
            axes[idx].set_yticks(np.arange(len(transformations)))
            axes[idx].set_yticklabels(transformations)
        else:
            axes[idx].set_yticks([])

        axes[idx].set_xticks(np.arange(len(norm_bounds)))
        axes[idx].set_xticklabels(norm_bounds)

        axes[idx].grid(False)
        for i in range(len(transformations) + 1):
            axes[idx].axhline(i - 0.5, color='black', linewidth=0.5)
        for j in range(len(norm_bounds) + 1):
            axes[idx].axvline(j - 0.5, color='black', linewidth=0.5)

    create_colorbar(fig, im)
    plt.suptitle(f'Mann-Whitney U Test Results Between Noise STDs 0.0 and {noise_std} for Different Figure Numbers on {args.experiment}', fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(filename, dpi=300)
    plt.close()

# 假设数据框已经加载
dfs = [
    df_original, df_gray, df_blue, df_top_black, df_diag_gray,
    df_center_red, df_center_black, df_center_noise
]

# 分别绘制并保存图片
plot_heatmap(0.05, dfs, f'mann_whitney_u_test_0.05_{args.experiment}.png')
plot_heatmap(0.1, dfs, f'mann_whitney_u_test_0.1_{args.experiment}.png')
plot_heatmap(0.2, dfs, f'mann_whitney_u_test_0.2_{args.experiment}.png')
# In[]
palette = sns.color_palette("bright")  # 您可以尝试不同的调色板，如"muted"、"bright"等
sns.set_style("whitegrid")
output_folder = 'output_plots'

def plot_ecdf(df, title_prefix, output_filename_prefix, show_legend=False):
    # plt.figure(figsize=(12,8.0))
    figure_num = 1600
    df[df['figure_num'] == figure_num]

    ax = sns.displot(
        data=df,
        y="norm",
        col="norm_bound",
        hue="injected_noise_std",
        kind="ecdf",
        aspect=0.75,
        linewidth=2,
        palette=palette,  # 使用Seaborn的颜色调色板
        height=2.5,
        log_scale=True,
        facet_kws=dict(margin_titles=False, sharey=True)
    )

    if show_legend:
        sns.move_legend(ax, loc="upper left", ncol=2, bbox_to_anchor=(0.10, 0.53))
        # sns.move_legend(ax, loc="upper right", ncol=2, bbox_to_anchor=(0.0, 1.22))
    else:
        ax._legend.remove()  # 如果不显示图例，则移除图例

    ax.set_ylabels('Norm discrepancy')
    ax.figure.suptitle(
        f"{title_prefix} - {args.experiment} - {args.activation} - {figure_num}figs", 
        y=0.92)
    figure_name = f"{output_filename_prefix}_{args.experiment}_{args.activation}_{figure_num}_ecdf.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, figure_name), dpi=800)
    plt.show()  # 显示图表

# 调用函数为每个DataFrame绘制图表，只有第一个调用时显示图例
plot_ecdf(df_gray, "No Distortion", "no distortion", show_legend=True)
plot_ecdf(df_gray, "Low Brightness Distortion", "low_brightness")
plot_ecdf(df_blue, "Blue distortion", "blue_distortion")
plot_ecdf(df_top_black, "Top black mask", "top_black_mask")
plot_ecdf(df_diag_gray, "Diagonal gray mask", "diag_gray_mask")
plot_ecdf(df_center_red, "Center red mask", "center_orange_mask")
plot_ecdf(df_center_black, "Center black mask", "center_black_mask")
plot_ecdf(df_center_noise, "Center noise mask", "center_noise_mask")
# In[]
# 定义一个函数来绘制小提琴图

palette = sns.color_palette("deep")  # 您可以尝试不同的调色板，如"muted"、"bright"等
sns.set_style("whitegrid")

output_folder = 'output_violins'

def plot_violin(df, title_prefix, output_filename_prefix, show_legend=False):
    # df = df[df['norm_bound'] == 32.0]
    figure_num = 200
    df = df[df['figure_num'] == figure_num]
    plt.figure(figsize=(5.0, 3.2))
    ax = sns.violinplot(
        data=df,
        x='norm_bound',
        y='norm',
        hue='injected_noise_std',
        inner="quart",
        bw_adjust=0.1,
        cut=1,
        linewidth=1,
        palette=palette,
        log_scale=(False, True),
        # width=1,
        # split=True,
        density_norm="count"
    )
    
    plt.title(f"{title_prefix} - {args.experiment} - {args.activation} - {figure_num}figs")
    plt.xlabel('Norm Bound')
    plt.ylabel('Norm Discrepancy')

    plt.tight_layout()
    if show_legend:
        plt.legend(title='Injected Noise Std')
        sns.move_legend(ax, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 0.6))
    else:
        plt.legend([], [], frameon=False)  # 隐藏图例
    figure_name = f"{output_filename_prefix}_{args.experiment}_{args.activation}_{figure_num}_violin.png"
    plt.savefig(os.path.join(output_folder, figure_name), dpi=500)
    plt.show()

# 调用函数为每个DataFrame绘制小提琴图
plot_violin(df_original, "No Distortion", "no_distortion", show_legend=True)
plot_violin(df_gray, "Low Brightness Distortion", "low_brightness")
plot_violin(df_blue, "Blue Distortion", "blue_distortion")
plot_violin(df_top_black, "Top Black Mask", "top_black_mask")
plot_violin(df_diag_gray, "Diagonal Gray Mask", "diag_gray_mask")
plot_violin(df_center_red, "Center Red Mask", "center_orange_mask")
plot_violin(df_center_black, "Center Black Mask", "center_black_mask")
plot_violin(df_center_noise, "Center Noise Mask", "center_noise_mask")
# In[]
def plot_violin_figure_num(df, title_prefix, output_filename_prefix, show_legend=False):
    bound = 32.0
    df = df[df['figure_num'] != 50]
    df = df[df['figure_num'] != 100]
    df = df[df['norm_bound'] == bound]
    plt.figure(figsize=(5.0, 3.2))
    ax = sns.violinplot(
        data=df,
        x='figure_num',
        y='norm',
        hue='injected_noise_std',
        inner="quart",
        bw_adjust=0.1,
        cut=1,
        linewidth=1,
        palette=palette,
        log_scale=(False, True),
        # width=1,
        # split=True,
        density_norm="area"
    )
    
    plt.title(f"{title_prefix} - {args.experiment} - {args.activation} - NormBound{bound}")
    plt.xlabel('Figure Num')
    plt.ylabel('Norm Discrepancy')

    plt.tight_layout()
    if show_legend:
        plt.legend(title='Injected Noise Std')
        sns.move_legend(ax, loc="lower right", ncol=2, bbox_to_anchor=(0.90, 0.1))
    else:
        plt.legend([], [], frameon=False)  # 隐藏图例
    figure_name = f"{output_filename_prefix}_{args.experiment}_{args.activation}_normbound{bound}_violin.png"
    plt.savefig(os.path.join(output_folder, figure_name), dpi=500)
    plt.show()

# 调用函数为每个DataFrame绘制小提琴图
plot_violin_figure_num(df_original, "No Distortion", "no_distortion", show_legend=True)
plot_violin_figure_num(df_gray, "Low Brightness Distortion", "low_brightness")
plot_violin_figure_num(df_blue, "Blue Distortion", "blue_distortion")
plot_violin_figure_num(df_top_black, "Top Black Mask", "top_black_mask")
plot_violin_figure_num(df_diag_gray, "Diagonal Gray Mask", "diag_gray_mask")
plot_violin_figure_num(df_center_red, "Center Orange Mask", "center_red_mask")
plot_violin_figure_num(df_center_black, "Center Black Mask", "center_black_mask")
plot_violin_figure_num(df_center_noise, "Center Noise Mask", "center_noise_mask")
# %%
def plot_transformed_images(transformed_images, scale_coefficient, save_path=None):
    # 解包转换后的图像
    original_img, gray_img, blue_img, top_black_img, top_noise_img, center_red_img, center_black_img, center_noise_img = transformed_images
    # 创建一个 2x4 的子图布局
    fig, axs = plt.subplots(2, 4, figsize=(8, 5.5))
    # 绘制各个图像
    images = [original_img, gray_img, blue_img, top_black_img, top_noise_img, center_red_img, center_black_img, center_noise_img]
    titles = ['Original Image', 'Low brightness', 'High blue', 'Top black mask', 'Diagonal gray mask', 'Center orange mask', 'Center black mask', 'Center uniform mask']
    for ax, img, title in zip(axs.flatten(), images, titles):
        ax.imshow(img.permute(1, 2, 0) / scale_coefficient)  # 假设图像通道顺序为 (C, H, W)
        ax.set_title(title)
        ax.axis('off')
    # 调整布局
    fig.tight_layout()
    # 如果提供了保存路径，则保存图像
    if save_path is not None:
        plt.savefig(save_path, dpi=500)
    # 显示图像
    plt.show()

referenced_file_name = "OAE_CIFAR10_Fig50_Norm_Bound4.0_H512_L7_E200000_LR0.001_OptAdam_Acttanh_SchCosine_Noise0.0_Seed0.pt"
args = parse_filename_to_args(referenced_file_name)
args.figure_num = 800
args.activation = "tanh"
args.experiment = 'TinyImageNet'
args.learning_rate = 7e-4
index = 16
args.noise_std = 0.20
args.norm_bound = 8.0

if args.experiment == 'ImageNet':
    args.figure_num = 100

trainset, input_size, img_size, scale_coefficient, img_indices = load_training_set(args)
imgs = trainset.data
model_filename = generate_model_filename(args)
model_path = os.path.join("models", model_filename)
model = torch.load(model_path).cuda()

output_path = "output_images"  # 指定保存图像的路径
os.makedirs(output_path, exist_ok=True)

save_path = f"{args.experiment}_{args.figure_num}.png"
imgs = transformations(imgs, img_size, scale_coefficient, index=index)
plot_transformed_images(imgs, scale_coefficient, save_path=os.path.join(output_path, save_path))

imgs = torch.stack(imgs)
imgs = imgs.cuda()

model_filename = generate_model_filename(args)
model_path = os.path.join("models", model_filename)
model = torch.load(model_path).cuda()

save_path = f"{args.experiment}_{args.activation}_NOAE_{args.noise_std}_{args.figure_num}.png"
queried_imgs = model.loop(imgs.reshape(imgs.size(0), -1), step_num=60).reshape(-1, *img_size).detach().cpu()
plot_transformed_images(queried_imgs, scale_coefficient, save_path=os.path.join(output_path, save_path))

args.noise_std = 0.0
model_filename = generate_model_filename(args)
model_path = os.path.join("models", model_filename)
model = torch.load(model_path).cuda()

save_path = f"{args.experiment}_{args.activation}_OAE_{args.noise_std}_{args.figure_num}.png"
queried_imgs = model.loop(imgs.reshape(imgs.size(0), -1), step_num=60).reshape(-1, *img_size).detach().cpu()
plot_transformed_images(queried_imgs, scale_coefficient, save_path=os.path.join(output_path, save_path))
# %%
