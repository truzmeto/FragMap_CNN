import numpy as np
import sys  
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))



def get_random_rotation(volume, target):
	'''
	Input: tensor
	Output: Randomly rotated tensor
	'''

	for i in range(3):
		temp = np.random.choice([2,3,4],2 ,replace = False)
		k = np.random.choice([1,2,3])
		rot_volume = torch.rot90(volume, int(k), [int(temp[0]), int(temp[1])])
		rot_target = torch.rot90(target, int(k), [int(temp[0]), int(temp[1])])

	return rot_volume, rot_target



#THIS IS NOT USED!
def get_24(tensor, is_gfe_map = False):
	"""
	> Values are repeated
	"""
	# print(tensor.shape)
	vol_rotated =[]
	for i in range(-4,4):
		for j in range(3):
			if j < 2 :
				temp = torch.rot90(tensor, int(i), [j+2, j+3])
			else:
				temp = torch.rot90(tensor, int(i), [j+2, j])
			# print(temp.shape)
			vol_rotated.append(temp)
	return vol_rotated
