import torch
import torch.nn as nn 
import torch.nn.functional as F

depth = 2

class SimpleLinear64(nn.Module):
    def __init__(self):
        super(SimpleLinear64, self).__init__()  

        self.lin = nn.Linear(64,144)
       
        self.lblock = nn.ModuleList()         
        self.lblock.append(nn.Linear(144,256))
        self.lblock.append(nn.Linear(256,144))
        self.lblock.append(nn.Linear(144,64))
        self.lblock.append(nn.Linear(64,36))
        self.lblock.append(nn.Linear(36,16))
        self.lblock.append(nn.Linear(16,8))
        self.lblock.append(nn.Linear(8,4))

        self.lout = nn.Linear(4,2)

    def forward(self, x):
        x = F.relu(self.lin(x))
        
        for j in range(7): 
            x = F.relu(self.lblock[j](x))
            y = x 

        x = self.lout(x)
        
        return x
    
    @torch.no_grad()
    def forward_evals(self, x):
        z = []
        x = F.relu(self.lin(x))
        z.append(x)
        for j in range(7): 
            x = F.relu(self.lblock[j](x))
            z.append(x)
        
        x = self.lout(x)
        z.append(x)

        return z