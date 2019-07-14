import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# based on unrolled gan pytoch
class MLP_G_(nn.Module):
   
    def __init__(self, z_dim=256, f_dim=128, ngpu=1, x_dim=2):
        super(MLP_G_, self).__init__()
        self.ngpu = ngpu
        self.all_layers = nn.ModuleList()
    
        layers = []
        layers.append(nn.Linear(z_dim, f_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(f_dim, f_dim))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(f_dim, x_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, z):
        if isinstance(z.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.main, z, range(self.ngpu))
        else:
            out = self.main(z)
        return out.view(out.size(0), -1)


class MLP_D_(nn.Module):

    def __init__(self, f_dim=128, ngpu=1, x_dim=2):
        super(MLP_D_, self).__init__()
        self.ngpu = ngpu
        self.x_dim = x_dim

        layers = []
        layers.append(nn.Linear(self.x_dim, f_dim))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(f_dim, 1))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            out = self.main(x)
        return out.squeeze()

