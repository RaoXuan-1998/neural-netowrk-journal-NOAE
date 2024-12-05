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

parser.add_argument('--data_type', type=str, default='tinyimagenet', help='random/mnist/svhn/cifar10/tinyimagenet')
parser.add_argument('--dim', type=int, default=160)
parser.add_argument('--figure_num', type=int, default=400, help='How many figures/random_points are used to train attractors')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--hidden_size', type=float, default=512, help='The number of hidden units in an MLP-like autoencoder')
parser.add_argument('--layer_num', type=float, default=7, help='The number of layers')
parser.add_argument('--epoch_num', type=int, default=200000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--mask_rate', type=float, default=1.00)
parser.add_argument('--uniform', type=float, default=False)
parser.add_argument('--max_noise_std', type=float, default=1.0)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--step_num', type=int, default=100)
parser.add_argument('--creterion', type=float, default=0.01)
parser.add_argument('--summarize_jacobians', type=bool, default=True)
parser.add_argument('--activation', type = str, default='sigmoid')
parser.add_argument('--model', type = str, default='noae')
parser.add_argument('--init', type = str, default=None)
parser.add_argument('--noise_std', type = float, default=0.05)
parser.add_argument('--max_noise_std_for_test', type = float, default = 5.0)
parser.add_argument('--k', type = int, default=5, help='default:k=5')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--bias', type=float, default=0.0)
parser.add_argument('--bound', type=float, default=1.2)
parser.add_argument('--xrange', type=int, default=6.0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)


def subspace_accuracy_visualization(model, trainset, device="cuda",
                                    name='radius_subspace_noise003_', xrange=args.xrange):
    model = model.to(device)
    loader = DataLoader(trainset, batch_size=len(trainset))
    X0 = iter(loader).next().to(device)
    x_list = [x for x in X0]
    
    V1 = []
    V2 = []
    V3 = []
    V4 = []
    
    V5 = []
    V6 = []
    
    V7 = []
    
    V512 = []
    
    V1_last = []
    V2_last = []

    with torch.no_grad():
        for index in tqdm.tqdm(range(len(X0))):
            x = x_list[index]
            jacobian = functional.jacobian(model, x)
            U,D,V = torch.linalg.svd(jacobian)
            values, indices = D.sort(descending=True)
            
            V1.append(V[0].unsqueeze(dim=0))
            V2.append(V[1].unsqueeze(dim=0))
            V3.append(V[2].unsqueeze(dim=0))
            V4.append(V[3].unsqueeze(dim=0))
            
            V5.append(V[4].unsqueeze(dim=0))
            V6.append(V[5].unsqueeze(dim=0))
            
            V7.append(V[6].unsqueeze(dim=0))
            
            V512.append(V[511].unsqueeze(dim=0))
            
            V1_last.append(V[-1].unsqueeze(dim=0))
            V2_last.append(V[-2].unsqueeze(dim=0))

    V1 = torch.cat(V1, dim=0)
    V2 = torch.cat(V2, dim=0)
    V3 = torch.cat(V3, dim=0)
    V4 = torch.cat(V4, dim=0)
    V5 = torch.cat(V5, dim=0)
    V6 = torch.cat(V6, dim=0)
    V7 = torch.cat(V7, dim=0)
    V512 = torch.cat(V512, dim=0)
    
    V1_last = torch.cat(V1_last, dim=0)
    V2_last = torch.cat(V2_last, dim=0)
    
    def evaluate(model, X0, first_vectors, second_vectors, x1, x2, creterion=0.01):
        X = X0 + x1*first_vectors + x2*second_vectors
        with torch.no_grad():
            X = model.loop(X, 5)
        norm = torch.norm(X - X0, dim=1)
        correct_index = norm < creterion
        correct_rate = correct_index.sum()/len(correct_index)
        return correct_rate.cpu().item()
    
    accs_all = []
    num = 81
    x1_list = torch.linspace(-xrange, xrange, num)
    x2_list = torch.linspace(-xrange, xrange, num)
    x1_grid, x2_grid = torch.meshgrid(x1_list, x2_list)
    grid = torch.cat(
        [x1_grid.reshape(-1).unsqueeze(1), x2_grid.reshape(-1).unsqueeze(1)], dim=1)
    
    accs = []
    for _ in tqdm.tqdm(range(len(grid))):
        (x1, x2) = grid[_]
        acc = evaluate(model, X0, V1, V2, x1, x2)
        accs.append(acc)
    accs = torch.tensor(accs).reshape(x1_grid.shape)
    accs_all.append(accs)
    
    accs = []
    for _ in tqdm.tqdm(range(len(grid))):
        (x1, x2) = grid[_]
        acc = evaluate(model, X0, V3, V4, x1, x2)
        accs.append(acc)
    accs = torch.tensor(accs).reshape(x1_grid.shape)
    accs_all.append(accs)

    accs = []
    for _ in tqdm.tqdm(range(len(grid))):
        (x1, x2) = grid[_]
        acc = evaluate(model, X0, V5, V512, x1, x2)
        accs.append(acc)
        
    accs = torch.tensor(accs).reshape(x1_grid.shape)
    accs_all.append(accs)
    
    accs = []
    for _ in tqdm.tqdm(range(len(grid))):
        (x1, x2) = grid[_]
        acc = evaluate(model, X0, V6, V1_last, x1, x2)
        accs.append(acc)
        
        
    accs = torch.tensor(accs).reshape(x1_grid.shape)
    accs_all.append(accs)

    fig, axs = plt.subplots(1, 4, figsize=(9.0, 2.2))
    figs = []
    for i, accs in enumerate(accs_all):
        ax = axs[i]
        if args.data_type == 'tinyimagenet':
            cs = ax.contourf(x1_grid, x2_grid, accs, levels=20, vmin=0.0, vmax=1.0, cmap=plt.get_cmap('magma'))
        else:
            cs = ax.contourf(x1_grid, x2_grid, accs, levels=20, vmin=0.0, vmax=1.0, cmap=plt.get_cmap('vorodos'))
        figs.append(cs)
        ax.set_xticks(torch.linspace(-0.8*xrange, 0.8*xrange, 3))
        if i == 0:
            ax.set_yticks(torch.linspace(-0.8*xrange, 0.8*xrange, 5))
        else:
            ax.set_yticks([])
            
        if i == 0:
            ax.set_xlabel(r'$\alpha_1$')
            ax.set_ylabel(r'$\alpha_2$', labelpad=0.2)
        elif i == 1:
            ax.set_xlabel(r'$\alpha_3$')
            ax.set_ylabel(r'$\alpha_4$', labelpad=0.2)
        elif i == 2:
            ax.set_xlabel(r'$\alpha_5$')
            ax.set_ylabel(r'$\alpha_{512}$', labelpad=0.2)
        elif i == 3:
            ax.set_xlabel(r'$\alpha_6$')
            ax.set_ylabel(r'$\alpha_{1024}$', labelpad=0.2)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.13) 
    cb = plt.colorbar(figs[0], ax=axs)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_label('Convergence accuracy')
    
    return fig
    
def subspace_basin_visualization(model, trainset, device="cuda",
                                 name='radius_subspace_noise003_', xrange=args.xrange,
                                 seperate=False):
    model = model.to(device)
    loader = DataLoader(trainset, batch_size=len(trainset))
    X0 = iter(loader).next().to(device)[0]

    with torch.no_grad():
        x = X0
        jacobian = functional.jacobian(model, x)
        U,D,V = torch.linalg.svd(jacobian)
        values, indices = D.sort(descending=True)
        
        v1 = V[0].unsqueeze(dim=0)
        v2 = V[1].unsqueeze(dim=0)
        
        v3 = V[2].unsqueeze(dim=0)
        v4 = V[3].unsqueeze(dim=0)
        
        v5 = V[4].unsqueeze(dim=0)
        v6 = V[5].unsqueeze(dim=0)
        
        v7 = V[6].unsqueeze(dim=0)
        v8 = V[7].unsqueeze(dim=0)
        
        v11 = V[9].unsqueeze(dim=0)
        
        v50 = V[49].unsqueeze(dim=0)
        v51 = V[50].unsqueeze(dim=0)
        
        v100 = V[99].unsqueeze(dim=0)
        v101 = V[100].unsqueeze(dim=0)

        v_last1 = V[-1].unsqueeze(dim=0)
        v_last2 = V[-2].unsqueeze(dim=0)

    def evaluate(model, X0, first_vectors, second_vectors, x1, x2, creterion=0.01):
        X = X0 + x1*first_vectors + x2*second_vectors
        with torch.no_grad():
            X = model.loop(X, 5)
        norm = torch.norm(X - X0, dim=1)
        correct_index = norm < creterion
        correct_rate = correct_index.sum()/len(correct_index)
        return correct_rate.cpu().item()

    accs_all = []
    num = 81
    x1_list = torch.linspace(-xrange, xrange, num)
    x2_list = torch.linspace(-xrange, xrange, num)
    x1_grid, x2_grid = torch.meshgrid(x1_list, x2_list)
    grid = torch.cat(
        [x1_grid.reshape(-1).unsqueeze(1), x2_grid.reshape(-1).unsqueeze(1)], dim=1)
    
    accs = []
    for _ in tqdm.tqdm(range(len(grid))):
        (x1, x2) = grid[_]
        acc = evaluate(model, X0, v1, v2, x1, x2)
        accs.append(acc)

    accs = torch.tensor(accs).reshape(x1_grid.shape)
    accs_all.append(accs)

    accs = []
    for _ in tqdm.tqdm(range(len(grid))):
        (x1, x2) = grid[_]
        acc = evaluate(model, X0, v3, v4, x1, x2)
        accs.append(acc)
    accs = torch.tensor(accs).reshape(x1_grid.shape)
    accs_all.append(accs)

    accs = []
    for _ in tqdm.tqdm(range(len(grid))):
        (x1, x2) = grid[_]
        acc = evaluate(model, X0, v5, v6, x1, x2)
        accs.append(acc)
        
    accs = torch.tensor(accs).reshape(x1_grid.shape)
    accs_all.append(accs)
    
    # accs = []
    # for _ in tqdm.tqdm(range(len(grid))):
    #     (x1, x2) = grid[_]
    #     acc = evaluate(model, X0, v7, v8, x1, x2)
    #     accs.append(acc)
        
    # accs = torch.tensor(accs).reshape(x1_grid.shape)
    # accs_all.append(accs)
    
    fig, axs = plt.subplots(1, 3, figsize=(6,2.2))
    for i, accs in enumerate(accs_all):
        ax = axs[i]
        cs = ax.contourf(x1_grid, x2_grid, accs, levels=2, cmap=plt.get_cmap('RdBu'))
        # ax.set_xticks(torch.linspace(-0.8*xrange, 0.8*xrange, 5))
        # ax.set_yticks(torch.linspace(-0.8*xrange, 0.8*xrange, 5))
        if i == 0:
            ax.set_xlabel(r'$l_1$')
            ax.set_ylabel(r'$l_2$', labelpad=0.05)
        elif i == 1:
            ax.set_xlabel(r'$l_3$')
            ax.set_ylabel(r'$l_4$', labelpad=0.05)
            ax.set_yticks([])
        elif i == 2:
            ax.set_xlabel(r'$l_{5}$')
            ax.set_ylabel(r'$l_{6}$', labelpad=0.05)
            ax.set_yticks([])
        elif i == 3:
            ax.set_xlabel(r'$l_{7}$')
            ax.set_ylabel(r'$l_{8}$', labelpad=0.05)
            ax.set_yticks([])
            
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.13) 
    
    return fig
    # cb = plt.colorbar(cs, ax=axs)
    # cb.set_ticks([0, 0.5, 1.0])
    # cb.set_label('Accuracy')
    
if args.data_type == "random":
	    exp_name = 'experiments/' + 'correlation-analysis-' + str(args.data_type) + '-{}samples'.format(args.figure_num) +\
	        '-{}dim'.format(args.dim) + '-{}layers'.format(args.layer_num) + '-{}widths'.format(args.hidden_size) +\
	            '-{}'.format(args.activation) + '-{}std'.format(args.noise_std) + '-{}bias'.format(args.bias) +\
                '-{}bound'.format(args.bound) + '-{}k'.format(args.k)
else:
	    exp_name = 'experiments/' + 'correlation-analysis-' + str(args.data_type) + '-{}samples'.format(args.figure_num) +\
	        	'-{}layers'.format(args.layer_num) + '-{}widths'.format(args.hidden_size) +\
	            '-{}'.format(args.activation) + '-{}std'.format(args.noise_std) +\
                '-{}'.format(args.k)
                
if args.data_type == "random":
    data_path = 'data/correlation-analysis/random-{}samples-{}dim-{}bound.pth'.format(
        args.figure_num, args.dim, args.bound)
    try:
        data = torch.load(data_path)
    except:
        "There is no dataset"

elif args.data_type == 'mnist':
    train_data = datasets.MNIST(
        root="../data/MNIST",  train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=True)
    label = 1
    count = 1
    data = []
    for index, (x, y) in enumerate(train_data):
        if (y+1)==label:
            data.append(x.view(-1))
            label=label+1
            if label == 11:
                label = label - 10
            count = count + 1
            if count > args.figure_num:
                break
    args.dim = 28*28
    
elif args.data_type == 'cifar10':
    train_data = datasets.CIFAR10(
        root="../data/",  train=True,
        transform=transforms.Compose(
            [transforms.Grayscale(num_output_channels=1), 
             transforms.ToTensor()]),
        download=True)
    label = 1
    count = 1 
    data = []
    for index, (x, y) in enumerate(train_data):
        if (y+1)==label:
            data.append(x.view(-1))
            label=label+1
            if label == 11:
                label = label - 10
            count = count + 1
            if count > args.figure_num:
                break
    args.dim = 32*32
    
elif args.data_type == 'tinyimagenet':
    train_data = datasets.ImageFolder(
        root='../data/tiny-imagenet-200/train',
        transform=transforms.Compose(
            [transforms.Resize([32, 32]),
             transforms.Grayscale(num_output_channels=1), 
             transforms.ToTensor()]
            ))
    label = 1
    indexs = torch.arange(0, len(train_data), 500)
    data = []
    for start_index in indexs:
        x = train_data[start_index][0]
        data.append(x.view(-1))
        if len(data) > args.figure_num:
            break
    args.dim = 32*32
    
elif args.data_type == 'svhn':
    train_data = datasets.SVHN(
        root="../data/SVHN",
        transform=transforms.Compose(
            [transforms.Grayscale(num_output_channels=1), 
             transforms.ToTensor()]),
        download=True)
    label = 1
    count = 1 
    data = []
    for index, (x, y) in enumerate(train_data):
        if (y+1)==label:
            data.append(x.view(-1))
            label=label+1
            if label == 11:
                label = label - 10
            count = count + 1
            if count > args.figure_num:
                break

    args.dim=32*32

trainset = Trainset(data)
model = torch.load(exp_name + '/model.pth')

fig = subspace_accuracy_visualization(model, trainset)

# subspace_basin_visualization(model, trainset)