# In[]
import torch
from utils import load_training_set, robustness_evaluation_Gaussian, robustness_evaluation_uniform
import argparse
import torch.nn.functional as F
import copy
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import cycle 
import random
import torch.nn as nn
# torch.cuda.set_device('cuda:5')

# In[]
parser = argparse.ArgumentParser()

parser.add_argument('--experiment', type=str, default='SVHN', help='MNIST, SVHN, CIFAR10, TinyImageNet')
parser.add_argument('--figure_num', type=int, default=400, help='How many figures are used to train attractors')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--creterion', type=float, default=0.01)

args = parser.parse_args(args=[])

class MCHN(object):
    def __init__(self, memories, beta=2.0, similarity_function='dot_product', use_kernel=False):
        super().__init__()
        self.memories = memories
        self.beta = beta
        self.similarity_function = similarity_function
        self.use_kernel = use_kernel
        
        if use_kernel:
            kernel, loss = train_kernel(memories, 200)
            self.kernel = kernel
        
    def Euclidean_distance(self, keys, queries):
        dist = torch.pow(keys.unsqueeze(1) - queries.unsqueeze(0), 2)
        return - dist.sum(dim=2)
    
    def dot_product_similarity(self, keys, queries):
        keys = F.normalize(keys, dim=1)
        queries = F.normalize(queries, dim=1)
        return keys @ queries.t()
    
    def Manhatten_distance(self, keys, queries):
        dist = torch.abs(keys.unsqueeze(1) - queries.unsqueeze(0))
        return - dist.sum(dim=2)
    
    def loop(self, queries, times=5):
        X = self.memories.to(queries.device)
        if self.use_kernel:
            X = self.kernel.to(queries.device)(X)
        for index in range(times):
            if self.use_kernel:
                queries = self.kernel.to(queries.device)(queries)

            if 'dot_product' in self.similarity_function:
                similarity = self.dot_product_similarity(X, queries)
            elif 'Manhatten' in self.similarity_function:
                similarity = self.Manhatten_distance(X, queries)
            elif 'Euclidean' in self.similarity_function:
                similarity = self.Euclidean_distance(X, queries)

            p = torch.softmax(self.beta * similarity, dim=0)

            queries = (self.memories.to(p.device).t() @ p).t()
        return queries
    
# [U-Hopfield]
class Kernel(nn.Module):
    def __init__(self, d):
        super(Kernel, self).__init__()
        self.w = nn.parameter.Parameter(torch.randn(2*d, d))

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return (self.w @ x.T).T
    
    def kernel_fn(self, u, v):
        with torch.no_grad():
            return self(u) @ self(v).T
    
def uniform_loss(x, t=2):
    x = F.normalize(x, dim=1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def train_kernel(memory, epoch):
    # print(memory.size())
    
    k = Kernel(memory.size(1)).cuda()

    opt = torch.optim.SGD(k.parameters(), lr=1)
    memory = memory.cuda()

    for i in range(epoch):
        opt.zero_grad()
        out = k(memory)
        loss = uniform_loss(out)
        loss.backward()
        opt.step()
        if i % 10 == 0:
          print( 'unif loss', round(loss.item(), 4))

    return k, loss.item()

# In[]
# # In[]
# mchn = MCHN(beta=100000000, similarity_function='dot_product')
# df = robustness_evaluation(mchn, imgs, max_noise_std=2.0)
# In[]
df_Gaussian_hopfield = {}
df_uniform_hopfield = {}
datasets = ['MNIST', 'SVHN', 'CIFAR10', 'TinyImageNet']

for exp in datasets:
    if exp  == 'TinyImageNet':
        args.figure_num = 200
    else:
        args.figure_num = 400
    dataset, input_size, img_size = load_training_set(args)
    imgs = torch.vstack(dataset.imgs)
    df_Gaussian_hopfield[exp] = {}
    df_uniform_hopfield[exp] = {}
    # for similarity_function in ['dot_product-kernel', 'Euclidean-kernel', 'Manhatten-kernel', 'dot_product', 'Euclidean', 'Manhatten']:
    for similarity_function in ['dot_product-kernel', 'dot_product']:
        if similarity_function == 'dot_product':
            beta = 1000.0
            use_kernel = False
        elif similarity_function == 'Euclidean':
            beta = 2.00
            use_kernel = False
        elif similarity_function == 'Manhatten':
            beta = 2.00
            use_kernel = False
        elif similarity_function == 'dot_product-kernel':
            beta = 1000.0
            use_kernel = True
        elif similarity_function == 'Euclidean-kernel':
            beta = 2.00
            use_kernel = True
        elif similarity_function == 'Manhatten-kernel':
            beta = 2.00
            use_kernel = True
            
        # key_Gaussian = (similarity_function, beta, 'Gaussian')
        # key_uniform = (similarity_function, beta, 'uniform')
        mchn = MCHN(imgs, beta=beta, similarity_function=similarity_function, use_kernel=use_kernel)
        df_Gaussian = robustness_evaluation_Gaussian(mchn, imgs, max_noise_std=2.0)
        df_uniform = robustness_evaluation_uniform(mchn, imgs)
        df_Gaussian_hopfield[exp][similarity_function] = df_Gaussian
        df_uniform_hopfield[exp][similarity_function] = df_uniform

# In[]
file_name_Gaussian = 'Hopfield-Gaussian.pth'
file_name_uniform = 'Hopfield-uniform.pth'
torch.save(df_Gaussian_hopfield, file_name_Gaussian)
torch.save(df_uniform_hopfield, file_name_uniform)

# In[]
