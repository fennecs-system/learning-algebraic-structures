import os
import numpy 
import itertools

import datetime
import time 

import typing 
import pickle 

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader

from simple_conv import SimpleConv 


def foo(x) :
    y = torch.matmul(x,x)
    return 0.1* torch.matmul(y,y) + 0.1 *torch.matmul(y,x) + y + 0.1 * x

def getData():
    f = open("cayleys.p","rb")
    cayleys = pickle.load(f)
    cayleys = list(map(lambda x: foo(x),cayleys))
    f.close()
    
    print(len(cayleys))
    cayleyClass = itertools.repeat(torch.Tensor([1,0]), 161280)

    f = open("latins.p","rb")
    latins = map(lambda x: foo(torch.nn.functional.normalize(torch.tensor(x))),pickle.load(f))
    f.close 

    latinClass = itertools.repeat(torch.Tensor([0,1]), 161280)

    latinData = [[l.unsqueeze(0),c] for l in latins for c in latinClass] 
    cayleyData = [[p.unsqueeze(0),c] for p in cayleys for c in cayleyClass]

    data = list(itertools.chain(latinData,cayleyData))
    return data


# train a simple model

model = SimpleConv()
data = getData()


length = len(data)
trainSize = int(0.8 * length)
valSize = length - trainSize

trainSplit, valSplit = torch.utils.data.random_split(data, [trainSize, valSize])

trainDataloader = DataLoader(trainSplit, batch_size=16, shuffle=True)
valDataloader = DataLoader(valSplit, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0005)

inputs, labels = next(enumerate(trainDataloader, 0))




startTime = time.time()

 # loop over the dataset 20 times

for epoch in range(50): 

    runningLoss = 0.0
    runningVal = 0.0

    for i, data in enumerate(trainDataloader, 0):

        inputs, labels = data
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        valIn, valLabel = next(iter(valDataloader))
        valOut = model(valIn)
        valLoss = criterion(valOut, valLabel)

        # print statistics
        runningLoss += loss.item()
        runningVal += valLoss.item()

        if i % 2000 == 1999:    # print every 2000 mini-batches
            
            totalTime = time.time() - startTime
            totalTimeStr = str(datetime.timedelta(seconds=int(totalTime))) 
            print(f'[{epoch + 1}, {i + 1:5d}] av loss: {runningLoss/2000.0} av val: {runningVal/2000.0} training time: {totalTimeStr}')
            runningLoss = 0.0
            runningVal = 0.0
    
torch.save(model.state_dict(), "model_conv.pt")
