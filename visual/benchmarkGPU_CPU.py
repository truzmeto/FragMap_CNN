#!/usr/bin/env python
import torch
import numpy as np
import timeit
import os
import matplotlib.pyplot as plt

##=====================================================================
plot_fn= '../figs/numpyVStorchCUDA.pdf'
fig = plt.figure(figsize=(5.,3.5))


#torch.set_default_tensor_type(torch.cuda.FloatTensor if 
#                              torch.cuda.is_available() else 
#                              torch.FloatTensor)

torch.set_default_tensor_type(torch.cuda.FloatTensor) 


def tGPU(x,y):
    
    x = x.cuda()
    y = y.cuda()
    n = 10
    t = 0
    for i in range(n):
        start = timeit.default_timer()
        z = torch.add(x,y)
        z = torch.exp(z)
        end = timeit.default_timer() 
        t = t + end - start
    return t / n

    
def tCPU(x,y):
    
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    n = 10
    t = 0
    for i in range(n):
        start = timeit.default_timer()
        z = np.exp(x + y)
        end = timeit.default_timer()
        t = t + end - start
    return  t / n
    

delta = 4
cpu2gpu = []
boxDim = []

for d in range(60, 106, delta):
    
    boxDim.append(d)
    x = torch.rand(3, 11, d, d, d)
    y = torch.rand(3, 11, d, d, d)
    t_cpu = tCPU(x,y)
    t_gpu = tGPU(x,y)
    cpu2gpu.append(t_cpu/t_gpu)


plt.plot(cpu2gpu, boxDim, 'o')
plt.plot(cpu2gpu, boxDim, '--')
plt.title("torchGPU vs numpy benchmarking")
plt.ylabel("Box Dim")
plt.xlabel(r"$\tau_{cpu}/\tau_{gpu}$")

fig.set_tight_layout(True)
plt.show()
fig.savefig(plot_fn)
os.system("epscrop %s %s" % (plot_fn, plot_fn))
