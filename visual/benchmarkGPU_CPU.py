#!/usr/bin/env python

import torch
import timeit
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.cuda.FloatTensor if 
                              torch.cuda.is_available() else 
                              torch.FloatTensor)

def tGPU(x,y):
    
    start_time = timeit.default_timer()
    x = x.cuda()
    y = y.cuda()
    z = torch.add(x,y)
    return timeit.default_timer() - start_time

    
def tCPU(x,y):
    
    start_time = timeit.default_timer()
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    z = x + y
    return timeit.default_timer() - start_time

    

d = 10
cpu2gpu = []
boxDim = []

for i in range(10):
    d = d + 10
    boxDim.append(d)
    x = torch.rand(3,11,d, d, d)
    y = torch.rand(3,11, d, d, d)
    t_cpu = tCPU(x,y)
    t_gpu = tGPU(x,y)
    cpu2gpu.append(t_cpu/t_gpu)


plt.plot(cpu2gpu, boxDim)
plt.title("torchGPU vs numpy benchmarking")
plt.ylabel("Box Dim")
plt.xlabel("cpuTime/gpuTime")
plt.show()

