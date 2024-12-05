# In[]
import argparse
import torch
from models import OAE
import os
import numpy as np
import random
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
# In[]
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

class Trainset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class Trainset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_training_set(args, indices=None):
    """
    Load and preprocess the training dataset.

    Args:
        args: Arguments containing experiment details.
            - seed (int): Random seed for reproducibility.
            - gray (bool): Whether to convert images to grayscale.
            - figure_num (int): Number of figures to use.
            - norm_scale (bool): Whether to normalize images to unit norm.
            - norm_bound (float): Bound for the norm of images after normalization.
        indices (list, optional): List of indices to use from the dataset. If provided, overrides `figure_num`.

    Returns:
        trainset (Trainset): Custom dataset containing preprocessed images.
        input_size (int): Size of the input features.
        img_size (tuple): Dimensions of the images.
        scale_coefficient (float): Scaling coefficient used for normalization.
    """
    torch.manual_seed(args.seed)
    
    if args.experiment == 'TinyImageNet':
        image_size = 64

    elif args.experiment == 'ImageNet':
        image_size = 224

    else:
        image_size = 32
    
    transform_list = [transforms.Resize([image_size, image_size]), transforms.ToTensor()]

    if args.gray:
        transform_list.insert(0, transforms.Grayscale(num_output_channels=1))
    
    transform = transforms.Compose(transform_list)
    
    X = []
    
    # Selecting dataset based on experiment type
    if args.experiment == 'MNIST':
        train_data = datasets.MNIST(
            root="../data/MNIST",  train=True,
            transform=transform,
            download=True)
        img_size = (1, image_size, image_size)
    elif args.experiment == 'SVHN':
        train_data = datasets.SVHN(
            root="../data/SVHN",
            transform=transform,
            download=True)
        img_size = (1, image_size, image_size) if args.gray else (3, image_size, image_size)

    elif args.experiment == 'CIFAR10':
        train_data = datasets.CIFAR10(
            root="../data/",  train=True,
            transform=transform,
            download=True)
        img_size = (1, image_size, image_size) if args.gray else (3, image_size, image_size)

    elif args.experiment == 'TinyImageNet':
        train_data = datasets.ImageFolder(
            root='../../data/tiny-imagenet-200/train',
            transform=transform)
        img_size = (1, image_size, image_size) if args.gray else (3, image_size, image_size)

    elif args.experiment == 'ImageNet':
        train_data = datasets.ImageFolder(
            root='/home/raoxuan/data/ImageNet-2012/train/',
            transform=transform)
        img_size = (1, image_size, image_size) if args.gray else (3, image_size, image_size)
    
    # Handling custom indices
    if indices is not None:
        if len(indices) < args.figure_num:
            raise ValueError("The number of required figure number exceeds the provided indices.")
        indices_to_use = indices[:args.figure_num]
    else:
        indices_to_use = torch.randperm(len(train_data))[:args.figure_num]
    
    for index in indices_to_use:
        x, _ = train_data[index]
        X.append(x.view(-1))
    
    input_size = image_size * image_size if args.gray or args.experiment == 'MNIST' else 3 * image_size * image_size

    X = torch.stack(X)

    if args.norm_scale:
        max_norm = max(torch.norm(X, p=2, dim=1))
        scale_coefficient = args.norm_bound / max_norm
        X = X * scale_coefficient
    else:
        scale_coefficient = 1.0
    
    trainset = Trainset(X)
    return trainset, input_size, img_size, scale_coefficient, indices_to_use
# In[]
def gray_distortion(imgs, gray_level):
    imgs_copy = gray_level * copy.deepcopy(imgs)
    return imgs_copy 
    
def red_distortion(imgs, red_level):
    imgs = copy.deepcopy(imgs)
    imgs[:,0,:,:] = red_level * imgs[:,0,:,:]
    return imgs

def blue_distortion(imgs, blue_level):
    imgs = copy.deepcopy(imgs)
    imgs[:,2,:,:] = blue_level * imgs[:,2,:,:]
    return imgs

def apply_color_distortion(imgs, color, level):
    if color == 'gray':
        return gray_distortion(imgs, level)
    elif color == 'red':
        return red_distortion(imgs, level)
    elif color == 'blue':
        return blue_distortion(imgs, level)
    else:
        raise ValueError("Unsupported color: {}".format(color))

def robustness_evaluation_color_variation(
        model, data, img_size, scale_coefficient, color, max_noise_std=0.8, times=3):
    X0 = data.cuda()
    levels = [0.3, 0.5, 0.7, 1.4]  # 这里可以自定义不同的级别列表
    Norms = []
    for level in tqdm(levels, desc='Processing different levels'):
        norms = []
        for _ in tqdm(range(times), desc='Repeating evaluation'):
            data_noise = apply_color_distortion(X0, color, level, scale_coefficient)
            X = model.loop(data_noise, step_num=60)
            norm = torch.norm(X.data - X0, dim=1).unsqueeze(dim=1) / scale_coefficient
            norms.append(norm)
        norms = torch.cat(norms, dim=1)
        Norms.append(norms.mean(dim=1).unsqueeze(dim=1))
    Norms = torch.cat(Norms, dim=1)

    results = {}
    results['levels'] = levels
    results['norms'] = Norms.cpu().numpy()
    return results

def uniform_noise_distortion(imgs, mask_rate, scale_coefficient=1.0):
    element_num = imgs.size(1)
    mask_num = int(mask_rate*element_num)
    indices = range(0, element_num)
    mask_indices = random.sample(indices, mask_num)
    data_mask = copy.deepcopy(imgs)
    data_mask[:, mask_indices] = scale_coefficient*data_mask[:, mask_indices].uniform_(0.0, 1.0)
    return data_mask

def robustness_evaluation_uniform(
        model, data, img_size, scale_coefficient, max_mask_rate=1.0, times=3):
    X0 = data.cuda()
    mask_rates = torch.linspace(0.0, max_mask_rate, 3)
    Norms = []
    for mask_rate in tqdm(mask_rates, desc='loop 1'):
        norms = []
        for time in tqdm(range(times), desc='loop 2'):
            data_mask = uniform_noise_distortion(X0, mask_rate, scale_coefficient)
            X = model.loop(data_mask, step_num=60)
            norm = torch.norm(X.data - X0, dim=1).unsqueeze(dim=1) / scale_coefficient
            norms.append(norm)
        norms = torch.cat(norms, dim=1)
        Norms.append(norms.mean(dim=1).unsqueeze(dim=1))
    
    Norms = torch.cat(Norms, dim=1)
    results = {}
    results['mask_rates'] = mask_rates.numpy()
    results['norms'] = Norms.cpu().numpy()
    return results

def diagonal_mask_distortion(imgs, img_size, mask_width, mask_type='black', scale_coefficient=1.0):
    # 确保输入是正确的形状
    imgs = imgs.reshape(-1, *img_size)
    width, height = img_size[1], img_size[2]
    
    # 创建数据副本以避免改变原始数据
    data_mask = copy.deepcopy(imgs)
    
    def get_mask_range(x, y, width, height, mask_width):
        # 计算点 (x, y) 到对角线的距离
        A = height - 1
        B = -(width - 1)
        C = 0
        distance = abs(A * x + B * y + C) / ((A**2 + B**2)**0.5)
        # 如果距离小于等于扭曲宽度的一半，则该点在扭曲范围内
        return distance <= mask_width / 2
    
    # 根据不同的mask类型应用扭曲
    if mask_type == 'red':
        color = [0.8, 0.3, 0.0]
    elif mask_type == 'black':
        color = [0.0, 0.0, 0.0]
    elif mask_type == 'gray':
        color = [0.7, 0.7, 0.7]
    elif mask_type == 'noise':
        pass  # 对于噪声，我们将直接生成随机数
    else:
        raise ValueError("Unsupported mask type. Choose from 'red', 'black', 'gray', or 'noise'.")
    
    for i in range(width):
        for j in range(height):
            if get_mask_range(i, j, width, height, mask_width):
                if mask_type != 'noise':
                    for c in range(img_size[0]):
                        data_mask[:, c, j, i] = color[c] * scale_coefficient
                else:
                    for c in range(img_size[0]):
                        data_mask[:, c, j, i] = scale_coefficient * torch.rand_like(data_mask[:, c, j, i])
    return data_mask

def top_mask_distortion(imgs, img_size, mask_rate, mask_type='black', scale_coefficient=1.0):
    # 确保输入是正确的形状
    imgs = imgs.reshape(-1, *img_size)
    height = img_size[1]
    mask_indice = int(height * mask_rate)
    
    # 创建数据副本以避免改变原始数据
    data_mask = copy.deepcopy(imgs)
    
    if mask_type == 'red':
        # 设置掩码区域为红色
        red_color = [0.8, 0.3, 0.0]  # 假设颜色值范围在 [0, 1]
        for c in range(img_size[0]):
            data_mask[:, c, :mask_indice, :] += red_color[c] * scale_coefficient
    
    elif mask_type == 'black':
        # 设置掩码区域为黑色
        black_color = [0.0, 0.0, 0.0]
        for c in range(img_size[0]):
            data_mask[:, c, :mask_indice, :] = black_color[c] * scale_coefficient

    elif mask_type == 'gray':
        black_color = [0.7, 0.7, 0.7]
        for c in range(img_size[0]):
            data_mask[:, c, :mask_indice, :] = black_color[c] * scale_coefficient

    elif mask_type == 'noise':
        for c in range(img_size[0]):
            data_mask[:, c, :mask_indice, :] = scale_coefficient * torch.rand_like(data_mask[:, c, :mask_indice, :])
    else:
        raise ValueError("Unsupported mask type. Choose from 'red', 'black', or 'noise'.")
    
    return data_mask

def robustness_evaluation_top_mask(
        model, data, img_size, scale_coefficient, times=3, mask_type='red'):
    X0 = data.cuda()
    mask_rates = [0.3, 0.5]
    Norms = []
    for mask_rate in tqdm(mask_rates, desc='loop 1'):
        norms = []
        for time in tqdm(range(times), desc='loop 2'):
            data_mask = top_mask_distortion(X0, img_size, mask_rate, mask_type, scale_coefficient)
            data_mask = data_mask.reshape(data_mask.size(0), -1)
            X = model.loop(data_mask, step_num=60)
            norm = torch.norm(X.data - X0, dim=1).unsqueeze(dim=1) / scale_coefficient
            norms.append(norm)
        norms = torch.cat(norms, dim=1)
        Norms.append(norms.mean(dim=1).unsqueeze(dim=1))
    
    Norms = torch.cat(Norms, dim=1)
    results = {}
    results['mask_rates'] = mask_rates.numpy()
    results['norms'] = Norms.cpu().numpy()
    return results

def center_random_mask(imgs, img_size, mask_ratio, mask_type='red', scale_coefficient=1.0):
    imgs = imgs.reshape(-1, *img_size)
    N, C, H, W = imgs.shape
    mask_height = int(H * mask_ratio)
    mask_width = int(W * mask_ratio)
    
    start_h = (H - mask_height) // 2
    start_w = (W - mask_width) // 2
    data_mask = copy.deepcopy(imgs)
    
    if mask_type == 'red':
        # 设置掩码区域为红色
        red_color = [0.8, 0.6, 0.1]  # 假设颜色值范围在 [0, 1]
        for c in range(C):
            data_mask[:, c, start_h:start_h+mask_height, start_w:start_w+mask_width] = red_color[c] * scale_coefficient
    
    elif mask_type == 'black':
        # 设置掩码区域为黑色
        black_color = [0.0, 0.0, 0.0]
        for c in range(C):
            data_mask[:, c, start_h:start_h+mask_height, start_w:start_w+mask_width] = black_color[c] * scale_coefficient
    
    elif mask_type == 'noise':
        # 设置掩码区域为白噪声
        for c in range(C):
            noise = torch.rand_like(data_mask[:, c, start_h:start_h+mask_height, start_w:start_w+mask_width])
            data_mask[:, c, start_h:start_h+mask_height, start_w:start_w+mask_width] = noise * scale_coefficient
    else:
        raise ValueError("Unsupported mask type. Choose from 'red', 'black', or 'noise'.")
    
    return data_mask

def robustness_evaluation_center_random_mask(
        model, data, img_size, scale_coefficient, max_mask_rate=0.5, times=3, mask_type='red'):
    """
    评估模型在不同掩码率下的鲁棒性。
    
    参数:
        model: 模型对象。
        data (torch.Tensor): 输入图像数据，形状为 (N, C, H, W)。
        img_size (tuple): 图像尺寸 (C, H, W)。
        scale_coefficient (float): 颜色或噪声的缩放系数。
        max_mask_rate (float, optional): 最大掩码率，默认为 0.5。
        times (int, optional): 每个掩码率下重复实验的次数，默认为 3。
        mask_type (str, optional): 掩码区域的颜色类型，可选值为 'red', 'black', 'noise'，默认为 'red'。
        
    返回:
        dict: 包含不同掩码率下的平均范数结果。
    """

    X0 = data.cuda()
    mask_rates = torch.linspace(0.3, max_mask_rate, 2)
    Norms = []
    for mask_rate in tqdm(mask_rates, desc='loop 1'):
        norms = []
        for time in tqdm(range(times), desc='loop 2'):
            data_mask = center_random_mask(X0, img_size, mask_rate, mask_type, scale_coefficient)
            data_mask = data_mask.reshape(data_mask.size(0), -1)
            
            X = model.loop(data_mask, step_num=60)
            norm = torch.norm(X.data - X0, dim=1).unsqueeze(dim=1) / scale_coefficient
            norms.append(norm)

        norms = torch.cat(norms, dim=1)
        Norms.append(norms.mean(dim=1).unsqueeze(dim=1))
    
    Norms = torch.cat(Norms, dim=1)
    results = {}
    results['mask_rates'] = mask_rates.numpy()
    results['norms'] = Norms.cpu().numpy()
    return results

def transform_to_imgs(data, img_size, scale_coefficient):
    return (data / scale_coefficient).reshape(-1, *img_size)

def generate_model_filename(args):
    """
    Generate a filename for the model based on the experiment conditions.

    Args:
        args: Arguments object with experiment details.
    
    Returns:
        filename (str): A string representing the model file name.
    """
    base_name = f"OAE_{args.experiment}"
    base_name += f"_Fig{args.figure_num}"
    base_name += "_Gray" if args.gray else ""
    base_name += "_Norm" if args.norm_scale else ""
    base_name += f"_Bound{args.norm_bound}" if args.norm_scale else ""
    base_name += f"_H{args.hidden_size}_L{args.layer_num}"
    base_name += f"_E{args.epoch_num}_LR{args.learning_rate}"
    base_name += f"_Opt{args.optimizer}_Act{args.activation}_Sch{args.scheduler}"
    base_name += f"_Noise{args.noise_std}"
    base_name += f"_Seed{args.seed}"
    base_name += ".pt"  # Model file extension
    return base_name

def display_images(data, start_idx, end_idx, cols=5):
    data = data.cpu()
    if start_idx >= end_idx:
        raise ValueError("start_idx must be less than end_idx")
    
    if start_idx < 0 or end_idx > len(data):
        raise ValueError("Indices are out of bounds")

    num_images = end_idx - start_idx
    rows = (num_images + cols - 1) // cols  # 计算行数

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.tight_layout()

    for i in range(rows):
        for j in range(cols):
            idx = start_idx + i * cols + j
            if idx < end_idx:
                image = data[idx]
                image = image.permute(1, 2, 0)
                ax = axes[i, j] if rows > 1 else axes[j]
                ax.imshow(image.numpy())
                ax.set_title(f'Index {idx}')
                ax.axis('off')
            else:
                ax = axes[i, j] if rows > 1 else axes[j]
                ax.axis('off')
    plt.show()

def train_OAE(model, train_data, args):
    model = model.to(args.device)
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, betas = (0.9,0.999))
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9)
        
    if args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.7)
    elif args.scheduler == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num, eta_min=0.0)
    
    x = train_data.to(args.device)
    mse = nn.MSELoss(reduction='mean')
    
    for epoch in tqdm(range(args.epoch_num)):         
        optimizer.zero_grad()  
        x_noise = copy.deepcopy(x) 
        if args.noise_std > 0.0:
            x_noise += args.noise_std*torch.randn_like(x_noise)
            
        y = model(x_noise)
        loss = mse(x, y)
        loss.backward()
        
        if epoch <= 50000:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        scheduler.step()
        
        if epoch % 10000 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')

def parse_filename_to_args(filename):
    args = get_args()
    name = filename.replace(".pt", "")
    parts = name.split("_")
    idx = 0
    while idx < len(parts):
        part = parts[idx]  
        if part.startswith("OAE_"):
            args.experiment = part[4:]
            idx += 1
        elif part.startswith("Fig"):
            args.figure_num = int(part[3:])
            idx += 1
        elif part == "Gray":
            args.gray = True
            idx += 1
        elif part == "Norm":
            args.norm_scale = True
            bound_part = parts[idx + 1]
            if bound_part.startswith("Bound"):
                bound_value = bound_part[5:]
                args.norm_bound = float(bound_value)
                idx += 2
            else:
                args.norm_bound = args.norm_bound
                idx += 1
        elif part.startswith("LR"):
            args.learning_rate = float(part[2:])
            idx += 1
        elif part.startswith("H"):
            args.hidden_size = int(part[1:])
            idx += 1
        elif part.startswith("L"):
            args.layer_num = int(part[1:])
            idx += 1
        elif part.startswith("E"):
            args.epoch_num = int(part[1:])
            idx += 1
        elif part.startswith("Opt"):
            args.optimizer = part[3:]
            idx += 1
        elif part.startswith("Act"):
            args.activation = part[3:]
            idx += 1
        elif part.startswith("Sch"):
            args.scheduler = part[3:]
            idx += 1
        elif part.startswith("Noise"):
            args.noise_std = float(part[5:])
            idx += 1
        elif part.startswith("Seed"):
            args.seed = int(part[4:])
            idx += 1
        else:
            idx += 1
    
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='MNIST', help='MNIST, SVHN, CIFAR10, TinyImageNet, ImageNet')
    parser.add_argument('--figure_num', type=int, default=200, help='How many figures are used to train attractors')
    parser.add_argument('--gray', type=bool, default=False)
    parser.add_argument('--norm_scale', type=bool, default=True)
    parser.add_argument('--norm_bound', type=float, default=16.0)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--hidden_size', type=int, default=512, help='The number of hidden units in an MLP-like autoencoder')
    parser.add_argument('--layer_num', type=int, default=7, help='The number of layers')
    parser.add_argument('--epoch_num', type=int, default=200000)
    parser.add_argument('--learning_rate', type=float, default=7e-4)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'sigmoid', 'cosid', 'relu', 'sind', 'softplus'])
    parser.add_argument('--scheduler', type=str, default='Cosine', choices=['Cosine', 'StepLR'])
    
    parser.add_argument('--noise_std', type=float, default=0.05, choices=[0.00, 0.01, 0.05])
    args = parser.parse_args(args=[])
    
    return args

def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
# In[]
def main(args):
    set_seed(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    trainset, input_size, img_size, scale_coefficient, img_indices = load_training_set(args)
    
    model = OAE(input_size, args.hidden_size, args.layer_num, args.activation).cuda()
    train_data = trainset.data.clone() 
    train_OAE(model, train_data, args)

    model_filename = generate_model_filename(args)
    model_path = os.path.join("models", model_filename)
    torch.save(model, model_path)

    # Save indices
    indices_filename = os.path.splitext(model_filename)[0] + '_indices.npy'
    indices_path = os.path.join("models", indices_filename)
    np.save(indices_path, img_indices)

    # Gaussian_results = robustness_evaluation_Gaussian(model, train_data, img_size, scale_coefficient)
    # uniform_results = robustness_evaluation_uniform(model, train_data, img_size, scale_coefficient)
    # mask_results = robustness_evaluation_top_mask(model, train_data, img_size, scale_coefficient)

    # imgs = transform_to_imgs(trainset.data, img_size, scale_coefficient)
    # display_images(imgs, 0,5)

    # imgs = top_mask_distortion(trainset.data, img_size, 0.3, scale_coefficient)
    # imgs = transform_to_imgs(imgs, img_size, scale_coefficient)
    # display_images(imgs, 0,5)

    # imgs = Gaussian_noise_distortion(trainset.data, 0.3, scale_coefficient)
    # imgs = transform_to_imgs(imgs, img_size, scale_coefficient)
    # display_images(imgs, 0,5)

    # imgs = uniform_noise_distortion(trainset.data, 0.3, scale_coefficient)
    # imgs = transform_to_imgs(imgs, img_size, scale_coefficient)
    # display_images(imgs, 0,5)
# In[]
if __name__ == '__main__':
    torch.cuda.set_device('cuda:3')
    args = get_args()
    # experiments = ['MNIST', 'SVHN', 'CIFAR10', 'TinyImageNet']
    activations = ["tanh"]
    experiments = ['CIFAR10', 'TinyImageNet']
    norm_bound_list = [4.0, 8.0, 12.0, 16.0]
    for activation in activations:
        for dataset in experiments:
            if dataset == 'TinyImageNet':
                args.figure_num = 100
            else:
                args.figure_num = 200
            for norm_bound in norm_bound_list:
                args.activation = activation
                args.experiment = dataset
                args.norm_bound = norm_bound
                args.learning_rate =7e-4
                args.noise_std = 0.0
                main(args)
# %%
