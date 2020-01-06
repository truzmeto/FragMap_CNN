#!/usr/bin/env python
import torch
import numpy as np
import timeit
import os
import matplotlib.pyplot as plt

##=====================================================================
plot_fn= '../figs/numpyVStorchCUDA.pdf'
fig = plt.figure(figsize=(6.,4.))


#torch.set_default_tensor_type(torch.cuda.FloatTensor if 
#                              torch.cuda.is_available() else 
#                              torch.FloatTensor)

torch.set_default_tensor_type(torch.cuda.FloatTensor) 


def tGPU(x,y,n_repeat):
    n = n_repeat
    x = x.cuda()
    y = y.cuda()
    t = 0
    for i in range(n):
        torch.cuda.synchronize()
        start = timeit.default_timer()
        x = torch.add(x,x)
        y = torch.exp(y)
        end = timeit.default_timer()
        t = t + end - start
    return t / n

    
def tCPU(x,y,n_repeat):
    
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    n = n_repeat
    start = timeit.default_timer()
    for i in range(n):
        x = x + x
        y = np.exp(y)

    end = timeit.default_timer()
    t = end - start
    return  t / n
    

delta = 5
cpu2gpu = []
boxDim = []
n = 20
for d in range(10, 95, delta):
    
    boxDim.append(d)
    x = torch.rand(4, 11, d, d, d)
    y = torch.rand(4, 6, d, d, d)
    
    t_cpu = tCPU(x,y,n)
    t_gpu = tGPU(x,y,n)
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
