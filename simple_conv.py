import torch
import torch.nn as nn 
import torch.nn.functional as F


class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()  

        self.c1 = nn.Conv2d(1,4,(2,2))
        self.mp1 = nn.MaxPool2d(2,1)
        
        self.c2 = nn.Conv2d(4,8,(2,2))
        self.c3 = nn.Conv2d(8,16,(2,2))

        self.l1 = nn.Linear(64,16)
        self.l2 = nn.Linear(16,2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.mp1(x)
        x = F.relu(self.c2(x))
        x = self.mp1(x)
        x = F.relu(self.c3(x))
        x = self.mp1(x)
        x = torch.flatten(x,1)
        x = F.relu(self.l1(x))
        x = self.l2(x)   
        x = self.sigmoid(x)
        return x 
    
    @torch.no_grad()
    def forward_evals(self, x):
        z = []

        x = F.relu(self.c1(x))
        x = self.mp1(x)
        z.append(x)

        x = F.relu(self.c2(x))
        x = self.mp1(x)
        z.append(x)

        x = F.relu(self.c3(x))
        x = self.mp1(x)
        z.append(x)
        return z

        
