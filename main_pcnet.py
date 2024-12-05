# In[]
import torch.nn as nn
import torch.nn.functional as F
import torch
from copy import deepcopy
from utils import load_training_set
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.optim as optim

# In[]
class PCN(nn.Module):
    def __init__(self, layer_num, input_size, hidden_size, activation_function):
        super().__init__()
        self.hidden_linears = nn.ModuleDict()
        self.hidden_linears['layer_0'] = nn.Linear(hidden_size, input_size)
        
        self.layer_num = layer_num
        for i in range(layer_num - 1, 0, -1):
            self.hidden_linears[f'layer_{i}'] = nn.Linear(hidden_size, hidden_size)
            
        self.last_layer_paras = nn.Parameter(0.001*torch.randn([hidden_size]), requires_grad=True)
        if activation_function == "ReLU":
            self.act_fun = nn.ReLU()
        elif activation_function == "Tanh":
            self.act_fun = nn.Tanh()
        elif activation_function == "Sigmoid":
            self.act_fun = nn.Sigmoid()
        elif activation_function == "Softplus":
            self.act_fun = nn.Softplus()
        else:
            raise ValueError(f"Unknown activation function: {activation_function}")
        
        self.energy_func = torch.nn.MSELoss()
        self.internal_states = None

    def initialize_internal_states(self, x0, random=True):
        states = nn.ParameterDict()
        batch_size = x0.size(0)
        with torch.no_grad():
            if not random:
                states[f'layer_{self.layer_num}'] = nn.Parameter(
                    self.last_layer_paras.unsqueeze(dim=0).repeat((batch_size,1)).detach().clone(), True)
                    
                for i in range(self.layer_num - 1, 0, -1):
                    states[f'layer_{i}'] = nn.Parameter(
                        self.hidden_linears[f'layer_{i}'](states[f'layer_{i+1}']).detach().clone(), True)
            else:
                states[f'layer_{self.layer_num}'] = nn.Parameter(
                    0.1*torch.randn_like(
                        self.last_layer_paras.unsqueeze(dim=0).repeat((batch_size,1))
                    ), True)
                    
                for i in range(self.layer_num - 1, 0, -1):
                    states[f'layer_{i}'] = nn.Parameter(
                    0.1*torch.randn_like(
                        self.hidden_linears[f'layer_{i}'](states[f'layer_{i+1}'])
                    ), True)               

        return states

    def predict(self, internal_states, frozen_states=False):
        preds = {}
        batch_size = internal_states['layer_1'].size(0)

        if frozen_states:
            preds[f'layer_{self.layer_num}']  = self.last_layer_paras.unsqueeze(dim=0).repeat((batch_size,1))
            for i in range(self.layer_num - 1, -1, -1):
                preds[f'layer_{i}'] = self.hidden_linears[f'layer_{i}'](internal_states[f'layer_{i+1}'].data)
        else:
            preds[f'layer_{self.layer_num}']  = self.last_layer_paras.unsqueeze(dim=0).repeat((batch_size,1))
            for i in range(self.layer_num - 1, -1, -1):
                preds[f'layer_{i}'] = self.hidden_linears[f'layer_{i}'](internal_states[f'layer_{i+1}'])
        
        return preds
    
    def train_batch(self, x0, weight_optimizer, state_lr=0.001, T=10, reinitialize_states=False, random_initialization=False):
        if reinitialize_states:
            internal_states = self.initialize_internal_states(x0, random_initialization).to(x0.device)
        else:
            if self.internal_states == None:
                internal_states = self.initialize_internal_states(x0, random_initialization).to(x0.device)
            else:
                internal_states = self.internal_states

        state_optimizer = torch.optim.SGD(
            internal_states.parameters(), lr=state_lr)
        
        for step in range(T):
            self.inference_step(internal_states, x0, state_optimizer)
            energy_loss, loss = self.training_step(internal_states, x0, weight_optimizer)
        
        self.internal_states = internal_states
        return energy_loss, loss, energy_loss + loss
    
    def inference_step(self, internal_states, x0, state_optimizer):
        state_optimizer.zero_grad()
        preds = self.predict(internal_states)
        energy_loss = 0.0
        for key in internal_states.keys():
            energy_loss += self.energy_func(preds[key], internal_states[key])
        loss = self.energy_func(preds['layer_0'], x0)
        overall_loss = energy_loss + loss
        overall_loss.backward()
        state_optimizer.step()
        return energy_loss.cpu().item(), loss.cpu().item()
    
    def training_step(self, internal_states, x0, weight_optimizer):
        weight_optimizer.zero_grad()
        preds = self.predict(internal_states, frozen_states=True)
        energy_loss = 0.0
        for key in internal_states.keys():
            energy_loss += self.energy_func(preds[key], internal_states[key])
        loss = self.energy_func(preds['layer_0'], x0)
        overall_loss = energy_loss + loss
        overall_loss.backward()
        nn.utils.clip_grad_norm_(self.hidden_linears.parameters(), 5.0)
        nn.utils.clip_grad_norm_(self.last_layer_paras, 5.0)
        weight_optimizer.step()
        return energy_loss.cpu().item(), loss.cpu().item()
    
    def memory_retrieval(self, queries, state_lr=0.01, round=25, T=200):
        for i in range(round):
            internal_states = self.initialize_internal_states(queries, random=False).to(queries.device)
            state_optimizer = torch.optim.SGD(
                internal_states.parameters(), lr=state_lr)
            for j in range(T):
                energy_loss, loss = self.inference_step(internal_states, queries, state_optimizer)
                if i == 0 or i == round//2 or i == round - 1:
                    if j == 0 or j == T//2 or j == T - 1:
                        print(f'Memory_retrieval on a batch, round {i}, step {j+1}, energy_loss {energy_loss:.6f}, loss {loss:.6f}')
            with torch.no_grad():
                queries = self.predict(internal_states)['layer_0']
        return queries
    
    def loop(self, queries):
        return self.memory_retrieval(queries)

def train_and_save_model(model, imgs, args, seed, dataset_name):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    weight_optimizer = optim.Adam(
        [{'params': model.hidden_linears.parameters()}, {'params': model.last_layer_paras}], 
        lr=args.weight_lr, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, T_max=args.epoch_num, eta_min=args.weight_lr_min)

    loss_list = []
    for epoch in range(args.epoch_num):

        energy_loss, loss, overall_loss = model.train_batch(
            imgs, weight_optimizer, T = args.T,
            reinitialize_states = args.reinitialize_states, random_initialization = args.random_initialization)
        
        scheduler.step()
        print(f'Dataset: {dataset_name}, Seed: {seed}, Epoch: {epoch+1}, Energy Loss: {energy_loss:.6f}, Loss: {loss:.6f}, Overall_loss: {overall_loss:.6f}')
        loss_list.append(loss)
    
    # 保存模型
    model_path = f'pcnet_models/{dataset_name}_seed{seed}.pt'
    torch.save(model, model_path)
    print(f'Model saved to {model_path}')
    
    # 绘制损失曲线
    plt.figure()
    plt.plot(loss_list)
    plt.yscale('log')
    plt.title(f'{dataset_name} Seed {seed} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Energy Loss')
    plt.savefig(f'pcnet_figures/{dataset_name}_seed{seed}_loss.png')
    plt.close()

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='TinyImageNet', help='MNIST, SVHN, CIFAR10, TinyImageNet')
    parser.add_argument('--figure_num', type=int, default=400, help='How many figures are used to train attractors')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--weight_lr', type=float, default=1e-3)
    parser.add_argument('--weight_lr_min', type=float, default=1e-3)
    parser.add_argument('--epoch_num', type=int, default=20000)
    parser.add_argument('--layer_num', type=int, default=4)
    parser.add_argument('--activation_function', type=str, default='Softplus')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--reinitialize_states', type=float, default=False)
    parser.add_argument('--random_initialization', type=float, default=False)
    parser.add_argument('--T', type=int, default=10)
    args = parser.parse_args(args=[])

    datasets = ['MNIST', 'SVHN', 'CIFAR10', 'TinyImageNet']
    seeds = [8]

    # # 创建必要的文件夹
    # os.makedirs('pcnet_models', exist_ok=True)
    # os.makedirs('pcnet_figures', exist_ok=True)

    # for dataset_name in datasets:
    #     for seed in seeds:
    #         args.experiment = dataset_name
    #         torch.manual_seed(seed)
    #         torch.cuda.manual_seed_all(seed)
    #         np.random.seed(seed)
    #         dataset, input_size, img_size = load_training_set(args)
    #         imgs = torch.vstack(dataset.imgs).cuda()
    #         model = PCN(args.layer_num, input_size, args.hidden_size, args.activation_function).cuda()
    #         train_and_save_model(model, imgs, args, seed, dataset_name)
    
    
    from tqdm import tqdm
    import copy
    import pandas as pd
    import random

    def robustness_evaluation_Gaussian(model, data, max_noise_std=1.0, times=1):
        X0 = data.cuda()
        noise_stds = torch.linspace(0.0, max_noise_std, 11)
        Norms = []
        for noise_std in tqdm(noise_stds, desc='loop 1'):
            norms = []
            for time in tqdm(range(times), desc='loop 2'):
                data_noise = copy.deepcopy(X0) + noise_std*torch.randn_like(X0).cuda()
                X = model.loop(data_noise)
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
                X = model.loop(data_mask)
                norm = torch.norm(X.data - X0, dim=1).unsqueeze(dim=1)
                norms.append(norm)
            norms = torch.cat(norms, dim=1)
            Norms.append(norms.mean(dim=1).unsqueeze(dim=1))
        
        Norms = torch.cat(Norms, dim=1)
        df = {}
        df['noise_stds'] = mask_rates.numpy()
        df['norms'] = Norms.cpu().numpy()
        return df

    datasets = ['MNIST', 'SVHN', 'CIFAR10', 'TinyImageNet']
    # model = torch.load('pcnet_models/TinyImageNet_seed1.pt').cuda()

    df_Gaussian_pcnet = {}
    df_uniform_pcnet = {}

    for exp in datasets:
        args.experiment = exp
        model = torch.load(f'pcnet_models/{exp}_seed8.pt').cuda()

        dataset, input_size, img_size = load_training_set(args)
        data = torch.vstack(dataset.imgs).cuda()

        df_Gaussian = robustness_evaluation_Gaussian(model, data, times=1)
        df_uniform = robustness_evaluation_uniform(model, data, times=1)

        df_Gaussian_pcnet[exp] = df_Gaussian
        df_uniform_pcnet[exp] = df_uniform

        torch.save(df_Gaussian_pcnet, 'pcnet_gaussian1.pth')
        torch.save(df_uniform_pcnet, 'pcnet_uniform1.pth')
        
# queires = 0.7*torch.randn_like(imgs) + imgs
# memories = model.loop(queires)
# norm = torch.norm(memories - imgs, dim=1).mean()
# In[]
# In[]
# %%
# import matplotlib.pyplot as plt
# import optuna
# import torch.optim as optim

# def objective(trial):
#     # Define the hyperparameters to optimize

#     hidden_size = trial.suggest_int("hidden_size", 128, 1024, step=128)  # Hidden size in range [128, 1024]
#     epoch_num = trial.suggest_int("epoch_num", 1000, 10000, step=1000)  # Number of epochs
#     weight_lr = trial.suggest_loguniform("weight_lr", 1e-5, 1e-2)  # Weight learning rate in log scale
#     weight_lr_min = trial.suggest_loguniform("weight_lr_min", 1e-6, 1e-3)  # Minimum learning rate
#     activation_function = trial.suggest_categorical(
#         "activation_function", ["ReLU", "Tanh", "Sigmoid", "Softplus"]
#     )

#     # Load dataset
#     dataset, input_size, img_size = load_training_set(args)
#     imgs = torch.vstack(dataset.imgs).cuda()

#     # Create the model
#     model = PCN(
#         layer_num=4, 
#         input_size=input_size, 
#         hidden_size=hidden_size, 
#         activation_function=activation_function  # Pass the activation function choice
#     ).cuda()

#     # Set optimizer
#     weight_optimizer = optim.Adam(
#         [{'params': model.hidden_linears.parameters()}, {'params': model.last_layer_paras}],
#         lr=weight_lr
#     )

#     # Training loop
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, T_max=epoch_num, eta_min=weight_lr_min)

#     loss_list = []
#     for epoch in range(epoch_num):
#         # Training on the batch
#         loss = model.train_batch(imgs.cuda(), weight_optimizer, reinitialize_states=False)
#         scheduler.step()

#         # Track loss for optimization
#         loss_list.append(loss)

#         # Print progress
#         if epoch % 100 == 0:
#             print(f"Epoch {epoch}, Loss: {loss}")

#     # Queries with noise to perturb the original images
#     queries = 0.7 * torch.randn_like(imgs) + imgs  # Add noise to images

#     # Retrieve memories using the model's loop method
#     memories = model.loop(queries.cuda())  # Memory retrieval step

#     # Compute the norm between memories and original images
#     norm = torch.norm(memories - imgs.cuda(), dim=1).mean()

#     # Return the norm as the objective to minimize
#     return norm.item()

# def run_optuna_optimization():
#     # Create an Optuna study to minimize the objective function
#     study = optuna.create_study(
#         direction="minimize", 
#         storage="sqlite:///example.db",  # Path to SQLite database file
#         study_name="pytorch_pcn_optimization",  # Name of the study
#         load_if_exists=True  # If study exists, load it
#     )
    
#     study.optimize(objective, n_trials=60)  # Number of trials (experiments)

#     # Print the best trial
#     print(f"Best trial: {study.best_trial.params}")
#     print(f"Best value: {study.best_value}")

#     # Visualize the optimization process
#     optuna.visualization.plot_optimization_history(study)
#     plt.show()

# if __name__ == "__main__":
#     run_optuna_optimization()
# %%

# %%
