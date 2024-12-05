import argparse
from utils import run_task, cal_jac_and_sum_rob, Trainset, sum_dynamic
from torchvision import datasets, transforms
import torch
from models import OAE
from tensorboardX import SummaryWriter
import os
import logging
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm
import copy
import pandas as pd

import torch.autograd.functional as functional

torch.cuda.set_device('cuda:5')

parser = argparse.ArgumentParser()

parser.add_argument('--experiment', type=str, default='numerical-experiment')
parser.add_argument('--bound', type=float, default=0.3)
parser.add_argument('--dim', type=int, default=50)
parser.add_argument('--figure_num', type=int, default=25, help='How many random points are used in the training set')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--hidden_size', type=float, default=256, help='The number of hidden units in an MLP-like autoencoder')
parser.add_argument('--layer_num', type=float, default=4, help='The number of MLP layers')
parser.add_argument('--epoch_num', type=int, default=50000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--noise_std', type=float, default=0.00)
parser.add_argument('--mask_rate', type=float, default=1.00)
parser.add_argument('--uniform', type=float, default=False)
parser.add_argument('--max_noise_std', type=float, default=1.0)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--step_num', type=int, default=100)
parser.add_argument('--creterion', type=float, default=0.01)
parser.add_argument('--summarize_jacobians', type=bool, default=True)

args = parser.parse_args()

def sample_from_radius_bound(radius, dim, nsamples):
    """
    Generates samples uniformly within a hypersphere of a given radius.
    c
    Parameters:
    radius (float): The radius of the hypersphere.
    dim (int): The dimensionality of the space.
    nsamples (int): The number of samples to generate.
    
    Returns:
    torch.Tensor: A tensor of shape (nsamples, dim) where each row is a point inside the hypersphere.
    """
    s = torch.rand(nsamples, dim)  # Generate random points in the unit cube
    norm = torch.norm(s, dim=1).unsqueeze(dim=1)  # Calculate the norm of each point
    x = s * radius / norm  # Scale each point so it lies on the surface of the sphere, then multiply by radius
    return x

def sample_from_uniform_bound(bound, dim, nsamples, normalize=False):
    """
    Generates samples uniformly within a hypercube of a given bound.
    
    Parameters:
    bound (float): The half-length of the side of the hypercube (the hypercube spans from -bound to +bound).
    dim (int): The dimensionality of the space.
    nsamples (int): The number of samples to generate.
    normalize (bool): Whether to normalize the output (not used in this function).
    
    Returns:
    torch.Tensor: A tensor of shape (nsamples, dim) where each row is a point inside the hypercube.
    """
    s = 2 * bound * torch.rand(nsamples, dim) - bound  # Generate random points in the hypercube
    return s

def Plot_2D(args, model, trainset, trajectory=False):
    """
    Visualizes the 2D dynamics of the model over a grid of test points, optionally including trajectories.
    
    Parameters:
    - args: Namespace containing configuration parameters such as radius and creterion.
    - model: Trained model capable of processing test data through its 'loop' method.
    - trainset: List of tensors representing the training points.
    - trajectory (bool): If True, visualizes the trajectories of test points; otherwise, only plots the final positions.
    
    Returns:
    - list: A list of numpy arrays, each corresponding to a class of test data points.
    """
    
    # Define color palettes for different elements in the plot
    color_list1 = ['salmon', 'bisque', 'seagreen', 'c', 'cornflowerblue',
                   'lightpink', 'sandybrown', 'yellowgreen', 'white']
    
    color_list2 = ['darkred', 'darkorange', 'darkgreen', 'teal', 'darkblue', 
                   'deeppink', 'sienna', 'darkgoldenrod']
    
    # Set scaling factor for the plot range
    scale = 2.0
    
    # Generate a grid of test data points
    x_range = torch.linspace(-scale*args.radius, scale*args.radius, 400)
    y_range = torch.linspace(-scale*args.radius, scale*args.radius, 400)
    x, y = torch.meshgrid(x_range, y_range)
    test_data = torch.cat([x.reshape(-1,1), y.reshape(-1,1)], dim=1).cuda()
    
    # Process test data through the model to find converged positions
    with torch.no_grad():
        converged_test_data = model.loop(test_data, step_num=200)
    
    # Categorize test data based on proximity to training points
    test_data_classes = []
    break_out_indice = (torch.ones(len(test_data)) > 0.0).cuda()
    for x in trainset:
        x = x.cuda()
        # Determine which test points are close to the current training point
        True_indice = (converged_test_data - x).norm(dim=1) <= args.creterion
        False_indice = (converged_test_data - x).norm(dim=1) > args.creterion
        break_out_indice = break_out_indice * False_indice
        test_data_classes.append(converged_test_data[True_indice])
    # Add remaining test points that did not match any training point
    test_data_classes.append(converged_test_data[break_out_indice])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6,4))
    for i, class_data in enumerate(test_data_classes):
        # Convert tensor data to numpy for plotting
        test_data_np = class_data.cpu().numpy()
        plt.scatter(test_data_np[:,0], test_data_np[:,1], c=color_list1[i], s=1.0)
    
    # If trajectory visualization is enabled
    if not trajectory:
        # Generate a coarser grid for trajectory visualization
        x_range = torch.linspace(-scale*args.radius, scale*args.radius, 200)
        y_range = torch.linspace(-scale*args.radius, scale*args.radius, 200)
        x, y = torch.meshgrid(x_range, y_range)
        test_data = torch.cat([x.reshape(-1,1), y.reshape(-1,1)], dim=1).cuda()
        
        # Process the coarse grid data through the model
        with torch.no_grad():
            next_test_data = model.loop(test_data, step_num=1)
        change = (next_test_data - test_data)
        
        # Prepare data for streamplot
        u = change.cpu()[:,0].reshape(x.shape).numpy()
        v = change.cpu()[:,1].reshape(x.shape).numpy()
        plt.streamplot(x.numpy(), y.numpy(), u, v, density=2.0, color="black", linewidth=0.4, arrowsize=0.7)
        
        # Highlight training points
        for i, train_point in enumerate(trainset):
            plt.scatter(train_point[0], train_point[1], c=color_list2[i], marker='*', s=120.0)
        
        # Save the plot
        plt.savefig('figures/2D.jpg', dpi=500)
    
    # Return categorized test data
    return test_data_classes

def summarize(
        args, model, trainset, step_num, device, times=20, norm_num=10):
    """
    Analyzes the model's dynamics by evaluating fixed points and robustness of attractors.
    
    Parameters:
    - args: Namespace containing configuration parameters.
    - model: The model to analyze.
    - trainset: The dataset used for analysis.
    - step_num: Number of steps to evolve the system for analysis.
    - device: Device to perform computations on.
    - times: Number of times to repeat the robustness test.
    - norm_num: Not used in this function.
    
    Returns:
    - tuple: Containing the attractor rate and a list of robustness rates under different noise levels.
    """
    
    # Move the model to the specified device
    model = model.to(device)
    
    # Load the entire trainset into memory
    loader = DataLoader(trainset, batch_size=len(trainset))
    X0 = iter(loader).next().to(device)
    
    # Evolve the initial state through the model's dynamics
    with torch.no_grad():
        X = copy.deepcopy(X0)
        X = model.loop(X, step_num)

    # Identify fixed points based on a criterion
    fixed_points = torch.norm(X - X0, dim=1) < args.creterion
    fixed_points = fixed_points.cpu()
    
    # Prepare for eigenvalue analysis
    x_list = [x for x in X0]
    max_eigs = []
    
    # Calculate the maximum absolute eigenvalue for each Jacobian
    with torch.no_grad():
        print('Now calculating the maximum norm of eigenvalues and the maximum singular value for each Jacobian')
        for index in tqdm.tqdm(range(len(X0))):
            x = x_list[index]
            jacobian = functional.jacobian(model, x)
                
            eigenvalues, _ = torch.eig(jacobian, eigenvectors=True)
            eigenvalues_abs = (eigenvalues**2).sum(dim=1).sqrt()
                
            _, indices = eigenvalues_abs.sort(descending=True)
            max_eigs.append(eigenvalues_abs[indices[0]].item())

        # Identify attractors based on the eigenvalues
        attractor = (torch.tensor(max_eigs) < 1.0) * fixed_points
        attractor_rate = sum(attractor) / len(attractor)
        
    # Evaluate the robustness of attractors against noise
    discrete_noise_std_list = [0.1, 0.5, 1.0, 2.0]
    print('Now summarizing the robustness of attractors')
    all_correct_rates = []
    for step, noise_std in enumerate(discrete_noise_std_list):
        correct_rates = []
        for _ in range(times):
            # Add noise to the initial conditions
            X_noise = copy.deepcopy(X0) + noise_std * torch.randn_like(X0)
            with torch.no_grad():
                X = model.loop(X_noise, step_num)
                norm = torch.norm(X - X0, dim=1)
                # Calculate the proportion of points that remain near the original attractor
                correct_rate = norm < args.creterion
                correct_rates.append((correct_rate.sum() / len(correct_rate)).cpu().item())
        # Average the robustness across repetitions
        all_correct_rates.append(torch.tensor(correct_rates).mean().item())
    
    # Return the attractor rate and robustness rates
    return attractor_rate.item(), all_correct_rates

def main(args):
    x = torch.tensor(
        [[ 0.6939,  0.1105],
         [-0.3777, -0.0766],
         [ 0.2885, -0.4496],
         [-0.2064,  0.7772],
         [ 0.0088, -0.1420],
         [ 0.0114,  0.3439],
         [ 0.3321,  0.2003],
         [-0.5105,  0.3955]]
        )
    
    x = sample_from_uniform_bound(args.bound, args.dim, args.figure_num)
    
    torch.save(x, 'test.pth')
    
    # x = torch.load('test.pth')

    trainset = Trainset(x)
    
    exp_name = 'Numerical-experiments-sigmoid-FigureNum_' + str(args.figure_num) + '-Dim_'+ str(args.dim) + '-Bound_' + str(args.bound) +\
                  '-NoiseSTD_' + str(args.noise_std) + '-MaskRate_' + str(args.mask_rate) +\
                '-LayerNum_' + str(args.layer_num) + '-Width_' + str(args.hidden_size)

    log_dir = os.path.join('logs', args.experiment, exp_name)
    
    summary_writer = SummaryWriter(log_dir)
    
    model = OAE(args.dim, args.hidden_size, args.layer_num)
    
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", args)
    
    args.batch_size = args.figure_num
    
    run_task(model, trainset, args.epoch_num, args.batch_size, 
             args.lr, args.device, args.noise_std, args.mask_rate,
             args.uniform, summary_writer)
    
    # attractor_rate, maximum_noise_tolerance = summarize(args, model, trainset, 100, "cuda", True)
    # return attractor_rate, maximum_noise_tolerance
    
    df1 = cal_jac_and_sum_rob(
        args, model, trainset, args.step_num,
        [0.00, 0.05, 0.1, 0.5, 1.0], args.creterion, device = 'cuda',
        singular = True, summarize=args.summarize_jacobians)
    
    df1.to_csv(log_dir + '\\fixed_point_evaluations.csv')
    
    summary_writer.close()
    
    torch.save(model, os.path.join(log_dir, 'oae_model.pth'))

if __name__ == '__main__':
    # dims = np.arange(20, 130, 20)
    # dim = 50
    # bounds = np.arange(0.2, 1.3, 0.2)
    # bounds = [30, 60]
    # epsilons = [0.01, 0.02, 0.05, 0.10]
    # epsilons = [0.10]
    # ps = [0.05, 0.20, 0.40, 0.80, 1.00]
    # for dim in dims:
    # for bound in bounds:
    # for epsilon in epsilons:
    #         for p in ps:
    #             args.noise_std = epsilon
    #             args.mask_rate = p
    #             # args.bound = bound
    #             main(args)
                
    # figure_nums = np.arange(25, 410, 25)
    # for figure_num in figure_nums:
    #     args.figure_num = figure_num
        main(args)
        
    # main(args)
    # df = pd.DataFrame()
    # bounds = np.arange(0.4, 0.6, 0.1)
    # dim_list = []
    # bound_list = []
    # attractor_list = []
    # noise_tolerance_list = []
    
    # noise_std_001 = []
    # noise_std_005 = []
    # noise_std_010 = []
    # noise_std_020 = []
    
    # for i, bound in enumerate(bounds):
    #     args.bound = bound
    
    #     attractor_rate, noise_tolerance_accs = main(args)
    #     bound_list.append(bound)
    #     attractor_list.append(attractor_rate)
    #     dim_list.append(args.dim)
    #     noise_std_001.append(noise_tolerance_accs[0])
    #     noise_std_005.append(noise_tolerance_accs[1])
    #     noise_std_010.append(noise_tolerance_accs[2])
    #     noise_std_020.append(noise_tolerance_accs[3])

    # df['bound'] = bound_list
    # df['dim'] = dim_list
    # df['attractor_rate'] = attractor_list
    # df['noise_std_001'] = noise_std_001
    # df['noise_std_005'] = noise_std_005
    # df['noise_std_010'] = noise_std_010
    # df['noise_std_020'] = noise_std_020
    
    # try:
    #     old_df = pd.read_csv('logs/bound_evaluation_noise000.csv')
    #     df = pd.concat([old_df, df], ignore_index=True)
    #     df.to_csv('logs/bound_evaluation_noise001.csv', index=False)
    # except:
    #     df.to_csv('logs/bound_evaluation_noise001.csv', index=False)
    
    # df = pd.DataFrame()
    # noise_stds = [0.00, 0.01, 0.03]
    # dims = np.arange(10, 110, 10)
    
    # std_list = []
    # dim_list = []
    # attractor_list = []
    # noise_tolerance_list = []
    
    # for i, dim in enumerate(dims):
    #     args.dim = dim
    #     for j, noise_std in enumerate(noise_stds):
    
    #         args.noise_std = noise_std
    #         attractor_rate, maximum_noise_tolerance = main(args)
            
                        
    #         std_list.append(noise_std)
    #         dim_list.append(dim)
    #         attractor_list.append(attractor_rate)
    #         noise_tolerance_list.append(maximum_noise_tolerance)
            
    # df['dim'] = dim_list
    # df['noise_std'] = std_list
    # df['attractor_rate'] = attractor_list
    # df['average_noise_tolerance'] = noise_tolerance_list
    
    # df.to_csv('logs/bound_evaluation_bound100.csv')