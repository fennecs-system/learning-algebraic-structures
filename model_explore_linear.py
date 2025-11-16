import torch 
import numpy 
import os 
import math

import matplotlib
import matplotlib.pyplot as plt


from simple_linear import SimpleLinear64


torch.set_printoptions(sci_mode=False,linewidth=100)

def reshape(t):
    dim = (t.size())[0]
    l = torch.nn.Unflatten(0,(int(math.sqrt(dim)), int(math.sqrt(dim))))
    out = l(t)
    return out

model = SimpleLinear64()
model_dict = torch.load("model_linear.pt")
model.load_state_dict(model_dict)

f=open("4.csv", 'r')
csv=numpy.genfromtxt(f, delimiter=",")
f.close()

cayley = torch.flatten((torch.nn.functional.normalize(
            torch.from_numpy(csv).type(torch.float32))))


print(model(cayley))

evals = model.forward_evals(cayley)



evals = list(map(lambda x : reshape(x), evals[:-3]))

l1 = torch.imag(torch.fft.rfftn(evals[0])).detach().numpy()
l2 = torch.imag(torch.fft.rfftn(evals[1])).detach().numpy()
l3 = torch.imag(torch.fft.rfftn(evals[2])).detach().numpy()

plt.figure()

fig, axs = plt.subplots(4, 1)

axs[0].matshow(reshape(cayley))
axs[1].matshow(l1)
axs[2].matshow(l2)
axs[3].matshow(l3)

plt.savefig('plot.png')

