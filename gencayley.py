import os 
import torch
import pickle
import numpy 
import itertools
import random 

def transPerm(x : torch.Tensor, perm : list) -> torch.Tensor:
    rand = bool(random.getrandbits(1))
    if rand:
        x=x[perm,:]
    else:
        x=x[:,perm]
    return x

def genCayleys() -> list[(torch.Tensor,torch.Tensor)]:
    cayleys = []
    for file in ["1.csv","2.csv","3.csv","4.csv"]:
        f=open(file, 'r')
        print(f)
        csv=numpy.genfromtxt(file, delimiter=",")
        f.close()
        cayleys.append(torch.from_numpy(csv).type(torch.float32))
        print(cayleys) 

    normalised = list(map(torch.nn.functional.normalize,cayleys))
    perms = list(itertools.permutations(range(8)))
    cayleyPerms = itertools.chain.from_iterable(map(lambda n: map(lambda p: transPerm(n,p), perms), normalised))
    
    out = list(cayleyPerms)
    return out 

f=open("cayleys.p", "wb")
pickle.dump(genCayleys(),f)
f.close()

