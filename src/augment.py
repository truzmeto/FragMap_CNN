import numpy as np
import sys  
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

'''
TODO:
> Scaling ??
> Translation
'''


def rotate_90(tensor):
	# Do transpose tensor.transpose(axes1,axes2)
	temp = torch.rot90(tensor, 1 , [2,3])
	return temp

def rotate_180(tensor):
	# Do flip tensor.flip(axes)
	return 0

def rotate_270(tensor):
	# Do 90 + 270 cumulative
	return 0

def get_24(tensor):
	vol_rotated =[]
	for i in range(-4,4):
		for j in range(3):
			if j < 2 :
				temp = torch.rot90(tensor, i, [j+2, j+3])
			else:
				temp = torch.rot90(tensor, i, [j+2, j])
			vol_rotated.append(temp)
	return vol_rotated
# def get_random_rotation(tensor):
# 	'''
# 	Input: tensor
# 	Output: Randomly rotated tensor
# 	'''
# 	# rot = tensor.clone()
# 	for i in range(2,5):
# 		if np.random.random() > 0.2:
# 			print(".", end="")
# 			tensor = tensor.flip(i)
# 		if np.random.random() > 0.5:
# 			print('+',end = "")
# 			i = np.random.choice([2,3,4])
# 			j = np.random.choice([2,3,4])
# 			tensor = tensor.transpose(int(i),int(j))
# 	print("")
# 	return tensor