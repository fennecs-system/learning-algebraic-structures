import torch 
import numpy 
import os 
import math
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
import pickle 

from simple_conv import SimpleConv


torch.set_printoptions(sci_mode=False,linewidth=100)


model = SimpleConv()
model_dict = torch.load("model_conv.pt")
model.load_state_dict(model_dict)

f=open("1.csv", 'r')
csv=numpy.genfromtxt(f, delimiter=",")
f.close()

label = torch.Tensor([1,0])

#f = open("latins.p","rb")
#latins = map(lambda x: torch.nn.functional.normalize(torch.tensor(x)),pickle.load(f))
#f.close 
#latinData = [l.unsqueeze(0) for l in latins] 



cayley = (torch.nn.functional.normalize(
            torch.from_numpy(csv).type(torch.float32)).unsqueeze(0))


criterion = nn.BCELoss()


classification = model(cayley.unsqueeze(0))


print(classification)

print(criterion(classification,label.unsqueeze(0)))


# 
exit()

evals = model.forward_evals(cayley.unsqueeze(0))

l1 = (evals[0]).detach().numpy()
l2 = (evals[1]).detach().numpy()
l3 = (evals[2]).detach().numpy()


plt.figure()

fig, axs = plt.subplots(4, 16)

plt.axis("off")   # turns off axes
plt.axis("tight")  # gets rid of white border
plt.axis("image")  # square up the image instead of filling the "figure" space
axs[0,0].matshow(cayley[0])
for j in range(0,16):
    axs[0,j].axis('off')

for j in range(0,16):
    if j < 4:
        axs[1,j].matshow(l1[0,j])
    axs[1,j].axis('off')


for j in range(0,16):
    if j < 8:
        axs[2,j].matshow(l2[0,j])
    axs[2,j].axis('off')

for j in range(0,16):
    axs[3,j].matshow(l3[0,j])
    axs[3,j].axis('off')


plt.savefig('plot.png')
