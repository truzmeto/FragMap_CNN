import numpy as np
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import torch
#%matplotlib widget

x = np.outer(np.linspace(-4, 4, 100), np.ones(100))
#x = np.outer(np.random.rand(100)*8-4, np.ones(100))
y = x.copy().T # transpose
fs = 14

#Out_GFE = x;  Target = y
thresh = 1.0


##################### ----with torch-----#####################################
#x = torch.from_numpy(x)
#y = torch.from_numpy(y)
#l1 = torch.abs(x - y)

#step1 = 0.5*(1.0 - torch.sign( x + y - 2*thresh))
#step2 = 0.5*(1.0 - torch.sign(-y + thresh))
#step3 = 0.5*(1.0 - torch.sign( x - thresh))

#step1p  = 0.5*(1.0 - torch.sign(-x - y + 2*thresh))
#step2p = 0.5*(1.0 - torch.sign(-x + thresh))
#step3p = 0.5*(1.0 - torch.sign( y - thresh))

#z1 = l1*step1
#z2 = -2*(x-thresh)*step1p*step2 *step3
#z3 = -2*(y-thresh)*step1p*step2p*step3p
#z = z1 + z2 + z3
#z = z.numpy()
################################################################

l1 = np.abs(x - y)
step1 = 0.5*(1.0 - np.sign( x + y - 2*thresh))
step2 = 0.5*(1.0 - np.sign(-y + thresh))
step3 = 0.5*(1.0 - np.sign( x - thresh))

step1p = 0.5*(1.0 - np.sign(-x - y + 2*thresh))
step2p = 0.5*(1.0 - np.sign(-x + thresh))
step3p = 0.5*(1.0 - np.sign(y - thresh))
#step4 = 0.5*(1.0 - np.sign(-x - y + 2*thresh))
z1 = l1*step1
z2 = -2*(x-thresh)*step2*step3*step1p
z3 = -2*(y-thresh)*step2p*step3p*step1p
z = z1 + z2 + z3


fig = plt.figure(figsize = (12, 8))
ax = plt.axes(projection = '3d')
plt.ylabel(r"$Output_{GFE}$", fontsize = fs)
plt.xlabel(r"$Target_{GFE}$", fontsize = fs)
ax.plot_surface(x, y, z, cmap = 'viridis', edgecolor = 'none')


ax.set_title('Fancy L1',fontsize=fs)
plt.show()

