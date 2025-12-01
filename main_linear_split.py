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

from simple_linear import SimpleLinear64 

from functools import reduce


def loadData():
    f = open("cayleys.p","rb")
    cayleys = pickle.load(f)
    f.close()

    f = open("latins.p","rb")
    latins = map(lambda x: torch.nn.functional.normalize(torch.tensor(x)),pickle.load(f))
    f.close 
    return (list(cayleys),list(latins))




def prepData(cayleys,latins):

    latinClass = itertools.repeat(torch.Tensor([0,1]), len(latins) )
    cayleyClass = itertools.repeat(torch.Tensor([1,0]), len(cayleys))

    latinData = [[torch.flatten(l),c] for l in latins for c in latinClass] 
    cayleyData = [[torch.flatten(p),c] for p in cayleys for c in cayleyClass]

    data = list(itertools.chain(latinData,cayleyData))
    return data


# train a simple model
def trainModel(data):
    model = SimpleLinear64()

    # output is (p_c,p_l)
    # where 
    # - p_c is confidence of cayley
    # - p_l is confidence of just latin

    length = len(data)
    trainSize = int(0.8 * length)
    valSize = length - trainSize

    trainSplit, valSplit = torch.utils.data.random_split(data, [trainSize, valSize])
    trainDataloader = DataLoader(trainSplit, batch_size=64, shuffle=True)
    valDataloader = DataLoader(valSplit, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    startTime = time.time()

    # loop over the dataset 20 times
    for epoch in range(0): 

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
    return model 
        


(cayleys,latins) = loadData()

# 161280 total items

chunks = 32256 
cayleyChunks = [cayleys[i:i+chunks] for i in range(0, len(cayleys), chunks)]
latinChunks = [latins[i:i+chunks] for i in range(0, len(latins), chunks)]

chunks = list(map(prepData,cayleyChunks,latinChunks))

models = list(map(trainModel,chunks))


params = list(map(lambda m: list(m.parameters()), models))

print(params)


# averges = reduce(lambda x, acc : acc+x, params, 0)
#torch.save(model.state_dict(), "model_linear.pt")