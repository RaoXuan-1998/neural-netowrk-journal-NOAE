import argparse
from utils import Trainset
from torchvision import datasets, transforms
import torch
from models import OAE, NOAE
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm
import copy
import pandas as pd
import torch.nn as nn
import torch.autograd.functional as functional
import os

parser = argparse.ArgumentParser()

parser.add_argument('--data_type', type=str, default='cifar', help='random/mnist/cifar10/tinyimagenet')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--hidden_size', type=float, default=512, help='The number of hidden units in an MLP-like autoencoder')
parser.add_argument('--layer_num', type=float, default=8, help='The number of layers')
parser.add_argument('--epoch_num', type=int, default=300000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--step_num', type=int, default=100)
parser.add_argument('--creterion', type=float, default=0.01)
parser.add_argument('--summarize_jacobians', type=bool, default=True)
parser.add_argument('--activation', type = str, default='softplus')
parser.add_argument('--model', type = str, default='noae')
parser.add_argument('--init', type = str, default=None)
parser.add_argument('--noise_std', type = float, default=0.0)
parser.add_argument('--max_noise_std_for_test', type = float, default = 2.0)
parser.add_argument('--k', type = int, default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--bias', type=float, default=0.0)
parser.add_argument('--radius', type=float, default = 1.7)

args = parser.parse_args()

def run_task(
        model, train_data, epochs, batch_size, lr, device):
    model = model.to(device)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num,
                                                           eta_min=0.0)
    
    train_loader = DataLoader(train_data, len(train_data), shuffle=True)
    x = iter(train_loader).next().to(device)
    
    for epoch in tqdm.tqdm(range(epochs)):
        optimizer.zero_grad()
        x = x.to(device)
        loss = model.loss(x)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        scheduler.step()
        if epoch % 10000 == 0:
            print("epoch:%d, mse_loss:%f" % (epoch, loss.cpu().item()))

def sample_from_uniform_bound(bound, dim, nsamples, bias):
    s = 2*bound*torch.rand(nsamples, dim) - bound + bias
    return s

def sample_from_radius_bound(radius, dim, nsamples):
    s =2*torch.rand(nsamples, dim) - 1
    norm = torch.norm(s, dim=1).unsqueeze(dim=1)
    x = s*radius / norm
    return x

def cal_jac_and_sum_rob(
        args, model, test_data, step_num, creterion, device,
        singular = True, times = 2000, norm_num = 10, summarize=True, max_noise_std=1.0):

    model = model.to(device)
    loader = DataLoader(test_data, batch_size=len(test_data))
    X0 = iter(loader).next().to(device)
    
    layer_num = args.layer_num
    hidden_size = args.hidden_size
    
    with torch.no_grad():
        X = copy.deepcopy(X0)
        X = model.loop(X, 1)
        
    norm = torch.norm(X - X0, dim=1)

    fixed_points = torch.norm(X - X0, dim=1) < creterion
    fixed_points = fixed_points.cpu()
    
    summarizer = {}
    
    df = pd.DataFrame(
        {'fixed_point' : fixed_points}
        )
    
    df['norm'] = norm.cpu().tolist()
    
    summarizer['layer_num'] = layer_num
    summarizer['hidden_size'] = hidden_size

    if len(test_data[0]) < norm_num:
        norm_num = len(test_data[0]) 
    
    if summarize:
        eigenvalue_norms_list = [[] for i in range(norm_num)]
        eigenvalue_norm_sum = []
        singular_list = [[] for i in range(norm_num)]
        singular_sum = []

        x_list = [x for x in X0]
        with torch.no_grad():
            print('Now calculating the maxium norm of eigenvalues and the maximum singular value for each jacobian')
            for index in tqdm.tqdm(range(len(X0))):
                x = x_list[index]
                jacobian = functional.jacobian(model, x)
                eigenvalues, eigenvectors = torch.eig(jacobian, eigenvectors=True)
                eigenvalues_abs = (eigenvalues**2).sum(dim = 1).sqrt()
                
                values, indices = eigenvalues_abs.sort(descending=True)
                eigenvalue_norm_sum.append(values.sum().item())

                for i in range(norm_num):
                    eigenvalue_norms_list[i].append(values[i].item())
                
                if singular:
                    U,D,V = torch.linalg.svd(jacobian)
                    values, indices = D.sort(descending=True)
                    
                    singular_sum.append(values.sum().item())
                    for i in range(norm_num):
                        singular_list[i].append(values[i].item())
        
        df['eig_norm_sum'] = eigenvalue_norm_sum

        for i in range(norm_num):
            df['eig_norm_{}'.format(i)] = eigenvalue_norms_list[i]
        
        attractors = torch.tensor(eigenvalue_norms_list[0]) < 1.0
        df['attractors'] = attractors*fixed_points
        
        if singular:
            df['singular_sum'] = singular_sum
            for i in range(norm_num):
                df['singular_{}'.format(i)] = singular_list[i]

        print('Now summarize the roubustness of attractors')
        noise_std_list = torch.linspace(0.0, max_noise_std, 501)
        
        max_noise_radius_list = []
        x_list = [x.unsqueeze(0) for x in X0]
        for x0 in x_list:
            x0 = x0.repeat(times, 1)
            max_noise_radius = 0.0
            for i in tqdm.tqdm(range(len(noise_std_list))):
                noise_std = noise_std_list[i]
                x_noise = copy.deepcopy(x0) + noise_std*torch.randn_like(x0)
                x = model.loop(x_noise, step_num)
                norm = torch.norm(x-x0, dim=1)
                correct = norm < creterion
                correct = correct.cpu().numpy().astype(int)
                correct_rate = correct.sum()/len(correct)
                if correct_rate > 0.90:
                    max_noise_radius = noise_std.item()
                else:
                    break
            max_noise_radius_list.append(max_noise_radius)

        df["max_noise_tolerance"] = max_noise_radius_list
            
    return df

def get_cmap(n, name='viridis'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def Plot_2D(args, model, trainset, trajectory=False, random_color = False):
    
        # color_list1 = ['salmon', 'bisque', 'seagreen', 'c', 'cornflowerblue',
        #               'lightpink', 'sandybrown', 'yellowgreen', 'white']
        # color_list2 = ['darkred', 'darkorange', 'darkgreen', 'teal', 'darkblue', 
        #                 'deeppink', 'sienna', 'darkgoldenrod']
        
        # if len(trainset) > 9:
        
        if len(trainset) <= 20:
            color_list1 = get_cmap(len(trainset), name='tab20')
            color_list2 = get_cmap(len(trainset), name='viridis')
        else:
            color_list1 = get_cmap(len(trainset), name='viridis')
            color_list2 = get_cmap(len(trainset), name='PuRd')
        
        scale = 2.0
        
        x_range = torch.linspace(-scale*args.radius, scale*args.radius, 400)
        y_range = torch.linspace(-scale*args.radius, scale*args.radius, 400)
        x, y = test_data = torch.meshgrid(x_range, y_range)
        test_data = torch.cat([x.reshape(-1,1), y.reshape(-1,1)], dim=1).cuda()
        
        with torch.no_grad():
            converged_test_data = model.loop(test_data, step_num=200)
        
        test_data_classes = []
        break_out_indice = (torch.ones(len(test_data)) > 0.0).cuda()
        for x in trainset:
            x = x.cuda()
            True_indice = (converged_test_data - x).norm(dim=1) <= args.creterion
            False_indice = (converged_test_data - x).norm(dim=1) > args.creterion
            break_out_indice = break_out_indice*False_indice
            test_data_classes.append(test_data[True_indice])
        test_data_classes.append(test_data[break_out_indice])
        # import random
        # random.shuffle(test_data_classes)
        
        # fig, ax = plt.subplots(figsize=(6,4))
        fig, ax = plt.subplots(figsize=(8,6))
        for i in range(len(test_data_classes)):
            test_data = test_data_classes[i].cpu().numpy()
            
            if i == len(test_data_classes) - 1:
                plt.scatter(test_data[:,0], test_data[:,1], color="white", s=1.0, alpha=1.0)
            else:
                plt.scatter(test_data[:,0], test_data[:,1], color=color_list1(i), s=1.0, alpha=1.0)
          
        if not trajectory:
            x_range = torch.linspace(-scale*args.radius, scale*args.radius, 200)
            y_range = torch.linspace(-scale*args.radius, scale*args.radius, 200)
            x, y = test_data = torch.tensor(np.meshgrid(x_range, y_range))
            
            test_data = torch.cat([x.reshape(-1,1), y.reshape(-1,1)], dim=1).cuda()
            
            with torch.no_grad():
                next_test_data = model.loop(test_data, step_num=1)
            change = 0.1*(next_test_data - test_data)
            
            # lw = change.norm(dim=1).cpu().reshape(x.shape).numpy()*1.5
            
            u = change.cpu()[:,0].reshape(x.shape).numpy()
            v = change.cpu()[:,1].reshape(x.shape).numpy()
            plt.streamplot(x.numpy(), y.numpy(), u, v, density=2.0, color="black", linewidth=0.4, arrowsize=0.7)
            
            for i in range(len(trainset)):
                x = trainset[i]
                plt.scatter(x[0], x[1], color=color_list2(i), marker='*', s=120.0)
                plt.savefig('figures/2D-cosid-0.02std-169samples.jpg', dpi=900)
        return test_data_classes

# data = torch.tensor(
#     [[ 0.6939,  0.1105],
#      [-0.3777, -0.0766],
#      [ 0.2885, -0.4496],
#      [-0.2064,  0.7772],
#      [ 0.0088, -0.1420],
#      [ 0.0114,  0.3439],
#      [ 0.3321,  0.2003],
#      [-0.5105,  0.3955]]
#     )

# data = torch.tensor([[-1.9380, -1.7445],
#         [-1.7145, -1.0902],
#         [-2.1004,  0.0278],
#         [-2.2268,  0.8027],
#         [-2.2640,  1.5013],
#         [-0.7156, -2.0327],
#         [-0.5421, -1.0780],
#         [-0.9593, -0.0854],
#         [-0.5512,  1.0932],
#         [-0.7887,  2.1131],
#         [ 0.9731, -2.1989],
#         [ 0.5839, -0.9699],
#         [ 0.7904, -0.2057],
#         [ 0.5371,  0.7444],
#         [ 0.2980,  2.1026],
#         [ 1.7835, -1.6557],
#         [ 1.8051, -0.9859],
#         [ 1.9916,  0.2264],
#         [ 2.0659,  0.4675],
#         [ 2.1440,  1.8301]])


x1 = torch.linspace(-3.0, 3.0, 8)
x2 = torch.linspace(-3.0, 3.0, 8)
x1, x2 = torch.meshgrid(x1, x2)

data = torch.cat([x1.reshape(-1, 1), x2.reshape(-1, 1)], dim=1)
data = data + 0.1*torch.randn_like(data)

# data = []
# for i in range(5):
#     x = sample_from_radius_bound(0.3*(i+1), 2, 5*(i+1))
#     data.append(x)

# data = torch.cat(data, dim=0)
plt.scatter(data[:,0], data[:,1])


# plt.scatter(data[:,0], data[:,1])

trainset = Trainset(data)

args.dim = 2

if args.model == 'oae':
    model = OAE(args.dim, args.hidden_size, args.layer_num, args.activation, init = args.init)

elif args.model == 'noae':
    model = NOAE(args.dim, args.hidden_size, args.layer_num,
                 args.activation, k = args.k, noise_std = args.noise_std, init = args.init)
    
args.batch_size = len(data)

run_task(model, trainset, args.epoch_num, args.batch_size, args.lr, args.device)

# attractor_rate, maximum_noise_tolerance = summarize(args, model, trainset, 100, "cuda", True)
# return attractor_rate, maximum_noise_tolerance

df = cal_jac_and_sum_rob(
    args, model, trainset, args.step_num, args.creterion, device = 'cuda',
    times = 2000, summarize=args.summarize_jacobians, max_noise_std=args.max_noise_std_for_test)

Plot_2D(args, model, trainset, trajectory=False, random_color = False)

df1 = df[df['max_noise_tolerance']>0.0]
print(np.corrcoef(df1['singular_0'], df1['max_noise_tolerance']))
print(np.corrcoef(df1['eig_norm_0'], df1['max_noise_tolerance']))
# plt.scatter(df1['singular_0'], df1['max_noise_tolerance'])