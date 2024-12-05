import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.nn as nn
import copy 
import matplotlib.pyplot as plt
import torch.autograd.functional as functional
import os
import pandas as pd
import random

class AverageMeter():
    def __init__(self,):
        self.sum = 0
        self.count = 0
    def update(self, val, count):
        self.sum += val*count
        self.count += count
    
    def average(self, ):
        return self.sum/self.count


def train_OAE(model, train_data, args, summary_writer=None):
    
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
    
    vector_size = x.shape[1]
    mask_num = int(args.mask_rate_for_noise_injection*vector_size)
    random_indices = range(0, vector_size)

    mse = nn.MSELoss(reduction='mean')
    
    for epoch in tqdm(range(args.epoch_num)):
        if args.noise_decay:
            decay_factor = (args.epoch_num - epoch) / args.epoch_num
        else:
            decay_factor = 1.0
            
        optimizer.zero_grad()
        
        mask_indices = random.sample(random_indices, mask_num)
        
        x_noise = copy.deepcopy(x)
        
        if args.noise_std > 0.0 and args.uniform_noise_bound == 0.0:
            x_noise[:, mask_indices] += (args.noise_std*decay_factor)*torch.randn_like(
                x_noise[:, mask_indices])
            
        elif args.uniform_noise_bound > 0.0 and args.noise_std == 0.0:
            x_noise[:, mask_indices] += x[:,mask_indices].uniform_(
                -args.uniform_noise_bound * decay_factor, args.uniform_noise_bound * decay_factor)
            
        y = model(x_noise)
        loss = mse(x, y)
        loss.backward()
        
        if epoch <= 50000:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        scheduler.step()
        
        if epoch % 10000 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')
        
        if summary_writer is not None:
            summary_writer.add_scalar('MSE_loss', loss.item(), epoch)


def robustness_evaluation_Gaussian(model, data, max_noise_std=1.0, creterion=0.01, times=50):
    X0 = data.cuda()
    correct_rate_means = []
    correct_rate_stds = []
    noise_stds = torch.linspace(0.0, max_noise_std, 11)
    with torch.no_grad():
        for noise_std in tqdm(noise_stds, desc='loop 1'):
            correct_rates = []
            for time in tqdm(range(times), desc='loop 2'):
                data_noise = copy.deepcopy(X0) + noise_std*torch.randn_like(X0).cuda()
                with torch.no_grad():
                    X = model.loop(data_noise, 20)
                norm = torch.norm(X - X0, dim=1).squeeze()
                correct = norm < creterion
                correct = correct.cpu()
                correct_rates.append(correct.sum().item()/correct.shape[0])
            
            correct_rates = torch.tensor(correct_rates)
            mean = torch.mean(correct_rates)
            std = torch.std(correct_rates)
            correct_rate_means.append(mean.item())
            correct_rate_stds.append(std.item())
    
    df = pd.DataFrame({
        'noise_stds' : list(noise_stds.numpy()),
        'correct_rate_means' : correct_rate_means,
        'correct_rate_stds' : correct_rate_stds
        })
    
    return df

def robustness_evaluation_uniform(model, data, max_mask_rate=1.0, creterion=0.01, times=50):
    X0 = data.cuda()
    correct_rate_means = []
    correct_rate_stds = []
    length = X0.shape[1]
    mask_rates = torch.linspace(0.0, max_mask_rate, 11)
    indices = range(0, length)
    with torch.no_grad():
        for mask_rate in tqdm(mask_rates, desc='loop 1'):
            correct_rates = []
            mask_num = int(mask_rate*length)
            for time in tqdm(range(times), desc='loop 2'):
                mask_indices = random.sample(indices, mask_num)
                data_mask = copy.deepcopy(X0)
                data_mask[:, mask_indices] = data_mask[:, mask_indices].uniform_(0.0, 1.0)
                with torch.no_grad():
                    X = model.loop(data_mask, 20)
                norm = torch.norm(X - X0, dim=1).squeeze()
                correct = norm < creterion
                correct = correct.cpu()
                correct_rates.append(correct.sum().item()/correct.shape[0])
                
            correct_rates = torch.tensor(correct_rates)
            mean = torch.mean(correct_rates)
            std = torch.std(correct_rates)
            correct_rate_means.append(mean.item())
            correct_rate_stds.append(std.item())
    
    df = pd.DataFrame({
        'mask_rates' : list(mask_rates.numpy()),
        'correct_rate_means' : correct_rate_means,
        'correct_rate_stds' : correct_rate_stds
        })
    return df

def cal_jac_and_sum_rob(
        args, model, test_data, step_num, creterion, device,
        singular = True, times = 20, norm_num = 10, summarize=True):
    
    model = model.to(device)
    loader = DataLoader(test_data, batch_size=len(test_data))
    X0 = iter(loader).next().to(device)
    
    layer_num = args.layer_num
    hidden_size = args.hidden_size
    noise_injection = args.noise_std > 0.0
    noise_inject_std = args.noise_std
    
    with torch.no_grad():
        X = copy.deepcopy(X0)
        X = model.loop(X, step_num)
        
    fixed_points = torch.norm(X - X0, dim=1) < creterion
    fixed_points = fixed_points.cpu()
    
    df = pd.DataFrame(
        {'fixed_point' : fixed_points}
        )
    
    df['layer_num'] = [layer_num]*len(fixed_points)
    df['hidden_size'] = [hidden_size]*len(fixed_points)
    df['noise_injection'] = [noise_injection]*len(fixed_points)
    df['noise_inject_std'] = [noise_inject_std]*len(fixed_points)
    
    if len(test_data[0]) < norm_num:
        norm_num = len(test_data[0]) 
    
    if summarize:
        eigenvalue_norms_list = [[] for i in range(norm_num)]
        eigenvalue_norm_sum = []
        singular_list = [[] for i in range(norm_num)]
        singular_sum = []
        normal_loss_list = []

        x_list = [x for x in X0]
        with torch.no_grad():
            print('Now calculating the maxium norm of eigenvalues and the maximum singular value for each jacobian')
            for index in tqdm.tqdm(range(len(X0))):
                x = x_list[index]
                jacobian = functional.jacobian(model, x)
                
                normal_loss = torch.norm(
                    torch.mm(jacobian, jacobian.transpose(0,1)) - torch.mm(jacobian.transpose(0,1), jacobian))
                
                eigenvalues, eigenvectors = torch.eig(jacobian, eigenvectors=True)
                eigenvalues_abs = (eigenvalues**2).sum(dim = 1).sqrt()
                
                values, indices = eigenvalues_abs.sort(descending=True)
                
                eigenvalue_norm_sum.append(values.sum().item())
                
                normal_loss_list.append(normal_loss.item())
                
                for i in range(norm_num):
                    eigenvalue_norms_list[i].append(values[i].item())
                
                if singular:
                    U,D,V = torch.linalg.svd(jacobian)
                    values, indices = D.sort(descending=True)
                    
                    singular_sum.append(values.sum().item())
                    for i in range(norm_num):
                        singular_list[i].append(values[i].item())
        
        df['eig_norm_sum'] = eigenvalue_norm_sum
        df['normal_loss'] = normal_loss_list
        
        for i in range(norm_num):
            df['eig_norm_{}'.format(i)] = eigenvalue_norms_list[i]
        
        attractors = torch.tensor(eigenvalue_norms_list[0]) < 1.0
        df['attractors'] = attractors
        
        if singular:
            df['singular_sum'] = singular_sum
            for i in range(norm_num):
                df['singular_{}'.format(i)] = singular_list[i]

        print('Now summarize the roubustness of attractors')
        maximum_noise_tolerance = torch.zeros(len(X0))
        index = 0
        noise_std_list = torch.linspace(0.0, 5.0, 501)
        for i in tqdm.tqdm(range(len(noise_std_list))):
            noise_std = noise_std_list[i]
            accumulated_correct_rates = torch.zeros(len(X0))
            
            for time in range(times):
                X_noise = copy.deepcopy(X0) + noise_std*torch.randn_like(X0)
                with torch.no_grad():
                    X = model.loop(X_noise, step_num)
                    norm = torch.norm(X - X0, dim=1)
                    correct_index = norm < creterion
                    accumulated_correct_rates[correct_index] += 1.0
                    
            accumulated_correct_rates =  accumulated_correct_rates/times
            fails = accumulated_correct_rates < 0.5
            fails = fails.cpu()
            
            for step, fail in enumerate(fails):
                if fail:
                    if maximum_noise_tolerance[step] > 0.0:
                        pass
                    else:
                        maximum_noise_tolerance[step] = noise_std
                else:
                    pass
    
            if all(fails):
                break

            df["max_noise_tolerance"] = maximum_noise_tolerance.tolist()
    return df

def sum_dynamic(args, model, test_data, step_num,
                max_noise_std, creterion, plot, device,
                log_dir=None, img_size=None, times=10):
    
    model = model.to(device)
    loader = DataLoader(test_data, batch_size=len(test_data))
    X0 = iter(loader).next().to(device)
    
    layer_num = args.layer_num
    hidden_size = args.hidden_size
    
    noise_std_list = torch.linspace(0.0, max_noise_std, 101)
    
    noise_injection = args.noise_std > 0.0
    noise_inject_std = args.noise_std
    
    correct_rate_list = []
    time_list = []
    noise_stds = []
    
    for time in range(times):
        for step, noise_std in enumerate(noise_std_list):
            X_noise = copy.deepcopy(X0) + noise_std*torch.randn_like(X0)
            with torch.no_grad():
                X = model.loop(X_noise, step_num)
            norm = torch.norm(X - X0, dim=1)
            correct = norm < creterion
            correct = correct.cpu()
            correct_rate = correct.sum().item() / correct.shape[0]
            correct_rate_list.append(correct_rate)
            time_list.append(time)
            noise_stds.append(noise_std.item())
    
    length = len(noise_stds)
        
    df = pd.DataFrame(
    {
        "layer_num" : [layer_num]*length,
        "hidden_size" : [hidden_size]*length,
        "noise_injection" : [noise_injection]*length,
        "noise_inject_std" : [noise_inject_std]*length,
        "correct_rate" : correct_rate_list,
        "noise_std" : noise_stds,
        "time" : time_list
        }
    )

    C,H,W = img_size
    
    noise_std_list_ = [0.0, 0.1, 0.5, 1.0]
    
    for noise_std in noise_std_list_:
        X_noise = copy.deepcopy(X0) + noise_std*torch.randn_like(X0)
        with torch.no_grad():
            X = model.loop(X_noise, step_num)
            fig, axs = plt.subplots(20,20, figsize=(14,20))
            axs = axs.reshape(-1)
            for index, x in enumerate(X0):
                axs[index].imshow(transforms.ToPILImage()(x.reshape(C,H,W)), cmap="Greys_r")
                axs[index].set_xticks([])
                axs[index].set_yticks([])
                if index == 399:
                    break
            if log_dir is not None:
                plt.savefig(os.path.join(log_dir, 'Samples.jpg'))
    
            fig, axs = plt.subplots(20,20, figsize=(14,20))
            axs = axs.reshape(-1)
            for index, x in enumerate(X_noise):
                axs[index].imshow(transforms.ToPILImage()(x.reshape(C,H,W)))
                axs[index].set_xticks([])
                axs[index].set_yticks([])
                if index == 399:
                    break
            if log_dir is not None:
                plt.savefig(os.path.join(log_dir, 'Samples-with-noises-{}.jpg'.format(noise_std)))
                
            fig, axs = plt.subplots(20,20, figsize=(14,20))
            axs = axs.reshape(-1)
            for index, x in enumerate(X):
                axs[index].imshow(transforms.ToPILImage()(x.reshape(C,H,W)))
                axs[index].set_xticks([])
                axs[index].set_yticks([])
                if index == 399:
                    break
            if log_dir is not None:
                plt.savefig(os.path.join(log_dir, 'Converaged-Samples-with-noises-{}.jpg'.format(noise_std)))
                
    return df
    

def test_dynamic(model, test_data, step_num, noise_std, creterion, device, img_size=None, times=10):
    model = model.to(device)
    loader = DataLoader(test_data, batch_size=len(test_data))
    X0 = iter(loader).next().to(device)

    correct_rates = []
    for time in range(times):
        X_noise = copy.deepcopy(X0) + noise_std*torch.randn_like(X0)
        with torch.no_grad():
            X = model.loop(X_noise, step_num)
        mse_error = torch.sum((X0 - X)**2, dim=1)
        correct = mse_error<creterion
        correct = correct.cpu()
        correct_rates.append(correct.sum().item()/correct.shape[0])
        
        if times == 1:
            return correct
        
        C,H,W = img_size
    return correct_rates


def train_limit_cycle(
        model, train_data, epochs, lr, group_num, device, noise_std, summary_writer=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.7)
    
    if len(train_data) & group_num == 0:
        batch_size = len(train_data) // group_num
    else:
        print('Wrong group num')

    train_loader = DataLoader(train_data, len(train_data))
    x0 = iter(train_loader).next()
    x0 = x0.reshape(group_num, batch_size, -1).to(device)
    y_label = torch.cat([x0[:,1:], x0[:,0].unsqueeze(dim=1)], dim=1)
    
    mse = nn.MSELoss(reduction='mean')
    for epoch in tqdm.tqdm(range(epochs)):
        average_meter = AverageMeter()
        optimizer.zero_grad()
        x_noise = x0 + noise_std*torch.randn_like(x0)
        y = model(x_noise)
            
        loss = mse(y_label, y)
        loss.backward()
        optimizer.step()
        average_meter.update(loss.item(), x0.shape[0])
        
        scheduler.step()
        if summary_writer is not None:
            summary_writer.add_scalar('MSE_loss', average_meter.average(), epoch)
            
def summarize_limit_cycle(
        model, train_data, group_num, device, creterion):
    
    model = model.to(device)

    if len(train_data) & group_num == 0:
        batch_size = len(train_data) // group_num
    else:
        print('Wrong group num')

    train_loader = DataLoader(train_data, len(train_data))
    x0 = iter(train_loader).next()
    x0 = x0.reshape(group_num, batch_size, -1).to(device)
    y_label = torch.cat([x0[:,1:], x0[:,0].unsqueeze(dim=1)], dim=1).reshape(len(train_data), -1)
    
    correct_num = 0
    with torch.no_grad():
        y = model(x0).reshape(len(train_data), -1)
        print(torch.norm(y - y_label, dim=1))
        correct = torch.norm(y - y_label, dim=1) < creterion
        correct = correct.cpu()
        correct_num += correct.sum().item()

    return correct_num / len(train_data)

class Trainset(Dataset):
    
    def __init__(self, imgs):
        self.imgs = imgs
        
    def __getitem__(self, index):
        return self.imgs[index]
    
    def __len__(self):
        return len(self.imgs)
    
def load_training_set(args, seed=0, gray=True, image_size=32):
    torch.manual_seed(seed) 
    transform_list = [transforms.Resize([image_size, image_size]), transforms.ToTensor()]
    if gray:
        transform_list.insert(0, transforms.Grayscale(num_output_channels=1))
    
    transform = transforms.Compose(transform_list)
    
    if args.experiment == 'MNIST':
        train_data = datasets.MNIST(
            root="../data/MNIST",  train=True,
            transform=transform,
            download=True)
        X = []
        indices = torch.randperm(len(train_data))[:args.figure_num]
        for index in indices:
            x, _ = train_data[index]
            X.append(x.view(-1))
        
        input_size = image_size * image_size
        img_size = (1, image_size, image_size) if gray else (1, image_size, image_size)
    
    elif args.experiment == 'SVHN':
        train_data = datasets.SVHN(
            root="../data/SVHN",
            transform=transform,
            download=True)
        X = []
        indices = torch.randperm(len(train_data))[:args.figure_num] 
        for index in indices:
            x, _ = train_data[index]
            X.append(x.view(-1))
        input_size = image_size * image_size if gray else 3 * image_size * image_size
        img_size = (1, image_size, image_size) if gray else (3, image_size, image_size)
    
    elif args.experiment == 'CIFAR10':
        train_data = datasets.CIFAR10(
            root="../data/",  train=True,
            transform=transform,
            download=True)
        X = []
        indices = torch.randperm(len(train_data))[:args.figure_num]
        for index in indices:
            x, _ = train_data[index]
            X.append(x.view(-1))
        input_size = image_size * image_size if gray else 3 * image_size * image_size
        img_size = (1, image_size, image_size) if gray else (3, image_size, image_size)
        
    elif args.experiment == 'TinyImageNet':
        train_data = datasets.ImageFolder(
            root='../../data/tiny-imagenet-200/train',
            transform=transform)
        
        X = []
        indices = torch.randperm(len(train_data))[:args.figure_num]
        for index in indices:
            x, _ = train_data[index]
            X.append(x.view(-1))
            
        input_size = image_size * image_size if gray else 3 * image_size * image_size
        img_size = (1, image_size, image_size) if gray else (3, image_size, image_size)
    
    trainset = Trainset(X)
    return trainset, input_size, img_size

# train_data = datasets.MNIST(root="../../data/MNIST",  train =True, transform=transforms.ToTensor(), download=True)
# stop = False
# flag = 1
# count = 1
# x_list = []
# for index, (x, y) in enumerate(train_data):
#     if (y+1) == flag:
#         x_list.append(x)
#         flag = flag + 1
#         if flag == 11:
#             flag = flag - 10
#             count = count + 1
#             if count > 200:
#                 break
# for index, x in enumerate(x_list):
#     x_list[index] = x.reshape(-1)
# trainset = Trainset(x_list)

# oae = OAE(28*28, 2048, 7)
# run_task(oae, trainset, 50000, 500, 1e-3, 'cuda', 0.1)

# correct_rate = max_eigenvalue_list = test_model(oae, trainset, 100, 0.2, 0.001, True, False, 'cuda')