import torch.nn as nn
import torch
import math

class Coisd(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x):
        return torch.cos(x) - x
    
class Sind(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(10*x)/5 + x

class OAE(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, layers:int,
                 activation:str, init = 'xavier'):
        super().__init__()
        
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(layers-3)])
        
        self.last_layer = nn.Linear(hidden_dim, input_dim)
        
        self.hidden_dim_sqrt = math.sqrt(hidden_dim)
        self.input_dim_sqrt = math.sqrt(input_dim)
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'cosid':
            self.activation = Coisd()
        elif activation == 'sind':
            self.activation = Sind()

        def xavier_uniform_init():
            nn.init.xavier_uniform_(self.first_layer.weight)
            for layer in self.hidden_layers:
                 nn.init.xavier_uniform_(layer.weight)
            nn.init.xavier_uniform_(self.last_layer.weight)
        
        def kaiming_uniform_init():
            nn.init.kaiming_uniform_(self.first_layer.weight)
            for layer in self.hidden_layers:
                 nn.init.kaiming_uniform_(layer.weight)
            nn.init.kaiming_uniform_(self.last_layer.weight)
            
        def sparse_init():
            nn.init.sparse_(self.first_layer.weight, sparsity=0.1)
            for layer in self.hidden_layers:
                 nn.init.sparse_(layer.weight, sparsity=0.1)
            nn.init.sparse_(self.last_layer.weight, sparsity=0.1)
            
        def normal_init():
            nn.init.normal_(self.first_layer.weight)
            for layer in self.hidden_layers:
                 nn.init.normal_(layer.weight)
            nn.init.normal_(self.last_layer.weight)
        
        self.init = init
        
        if init is not None:
            if init == 'xavier':
                xavier_uniform_init()
            elif init == 'kaiming':
                kaiming_uniform_init()
            elif init == 'sparse':
                sparse_init()
            elif init == 'normal':
                normal_init()
                
        self.mse = nn.MSELoss()
        
    def forward(self, x):
        x = self.first_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.last_layer(x)
        return x
    
    def loss(self, x0):
        x = self.forward(x0)
        loss = self.mse(x0, x)
        return loss
    
    def loop(self, x, step_num):
        for step in range(step_num):
            x = self.forward(x)
        return x
    
class NOAE(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, layers:int,
                 activation:str, init = 'xavier', k=5, noise_std=0.1):
        super().__init__()
        
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(layers-3)])
        
        self.last_layer = nn.Linear(hidden_dim, input_dim)
        
        self.hidden_dim_sqrt = math.sqrt(hidden_dim)
        self.input_dim_sqrt = math.sqrt(input_dim)
        
        self.k = k
        self.noise_std = noise_std

        if activation== 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation== 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'cosid':
            self.activation = Coisd()
        elif activation == 'sind':
            self.activation = Sind()

        def xavier_uniform_init():
            nn.init.xavier_uniform_(self.first_layer.weight)
            for layer in self.hidden_layers:
                 nn.init.xavier_uniform_(layer.weight)
            nn.init.xavier_uniform_(self.last_layer.weight)
        
        def kaiming_uniform_init():
            nn.init.kaiming_uniform_(self.first_layer.weight)
            for layer in self.hidden_layers:
                 nn.init.kaiming_uniform_(layer.weight)
            nn.init.kaiming_uniform_(self.last_layer.weight)
            
        def sparse_init():
            nn.init.sparse_(self.first_layer.weight, sparsity=0.1)
            for layer in self.hidden_layers:
                 nn.init.sparse_(layer.weight, sparsity=0.1)
            nn.init.sparse_(self.last_layer.weight, sparsity=0.1)
            
        def normal_init():
            nn.init.normal_(self.first_layer.weight)
            for layer in self.hidden_layers:
                 nn.init.normal_(layer.weight)
            nn.init.normal_(self.last_layer.weight)
        
        self.init = init
        
        if init is not None:
            if init == 'xavier':
                xavier_uniform_init()
            elif init == 'kaiming':
                kaiming_uniform_init()
            elif init == 'sparse':
                sparse_init()
            elif init == 'normal':
                normal_init()
        
        self.mse = nn.MSELoss()
        
    def forward(self, x):
        x = self.first_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.last_layer(x)
        return x
    
    def loss(self, x0):
        x0 = x0.repeat(self.k ,1)
        x = x0 + self.noise_std*torch.randn_like(x0)
        x = self.forward(x)
        loss = self.mse(x0, x)
        return loss
        
    def loop(self, x, step_num):
        for step in range(step_num):
            x = self.forward(x)
        return x
    
# class ConvOAE(nn.Module):
#     def __init__(self, dataset:int, hidden_size:int, n_hidden_layers:int):
#         super(ConvOAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 20, 4, stride=2, padding=1),
#             nn.SiLU(),
#             nn.Conv2d(20, 36, 4, stride=2, padding=1),
#             nn.SiLU(),
# 			nn.Conv2d(36, 48, 4, stride=2, padding=1),
#             nn.SiLU())
        
#         if dataset == "CIFAR10":
#             input_size = 48*4*4
#         elif dataset == "TinyImageNet":
#             input_size = 48*8*8

#         layers = [ResidLayer(input_size, hidden_size)] + \
#             [ResidLayer(hidden_size, hidden_size) for _ in range(n_hidden_layers)] + \
#                   [nn.Linear(hidden_size, input_size)]
#         self.mlp_layers = nn.ModuleList(layers)

#         self.decoder = nn.Sequential(
# 			nn.ConvTranspose2d(48, 36, 4, stride=2, padding=1),
#             nn.SiLU(),
# 			nn.ConvTranspose2d(36, 20, 4, stride=2, padding=1),
#             nn.SiLU(),
#             nn.ConvTranspose2d(20, 3, 4, stride=2, padding=1),
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         b, c, h, _ = x.size()
#         x = x.view(b, -1)
#         for layer in self.mlp_layers:
#             x = layer(x)
#         x = self.decoder(x.view(b, c, h, h))
#         return x