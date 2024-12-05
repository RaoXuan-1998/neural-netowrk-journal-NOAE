# In[]
import torch
import multiprocessing as mp
from main_norm import get_args, main

def run_experiment(gpu_id, activation, dataset, norm_bound, noise_std, figure_num):
    torch.cuda.set_device(f'cuda:{gpu_id}')
    args = get_args()
    args.activation = activation
    args.experiment = dataset
    args.norm_bound = norm_bound
    args.learning_rate = 7e-4
    args.noise_std = noise_std
    args.figure_num = figure_num
    
    main(args)

def run():
    activations = ["tanh"]
    experiments = ['CIFAR10', "SVHN"]
    figure_nums = [50, 100, 200, 400, 800, 1600]
    norm_bound_list = [4.0, 8.0, 16.0, 32.0]
    noise_std_list = [0.0, 0.05, 0.1, 0.2]
    
    # 假设有6个可用的GPU
    available_gpus = [0, 1, 2, 3, 4, 5]
    
    # 设置最大进程数
    max_processes = 12
    
    # 创建进程池
    with mp.Pool(max_processes) as pool:
        jobs = []
        for activation in activations:
            for dataset in experiments:
                for norm_bound in norm_bound_list:
                    for noise_std in noise_std_list:
                        for figure_num in figure_nums:
                            gpu_id = available_gpus[len(jobs) % len(available_gpus)]
                            job = pool.apply_async(run_experiment, (gpu_id, activation, dataset, norm_bound, noise_std, figure_num))
                            jobs.append(job)
        
        # 等待所有任务完成
        for job in jobs:
            job.wait()

if __name__ == '__main__':
    run()
# In[]
# %%
