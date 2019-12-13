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


def rotate_90(tensor, axes=[2,3]):
	# Flips along axis (2,3)
	temp = torch.rot90(tensor, 1 , axes)
	return temp

def rotate_180(tensor, axes=[2,3]):
	# Flips along axis (2,3)
	temp = torch.rot90(tensor, 2 , axes)
	return temp

def rotate_270(tensor, axes=[2,3]):
	# Flips along axis (2,3)
	temp = torch.rot90(tensor, 3 , axes)
	return temp

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


def get_random_rotation(volume, target):
	'''
	Input: tensor
	Output: Randomly rotated tensor
	'''
	# rot = tensor.clone()
	for i in range(3):
		temp = np.random.choice([2,3,4],2 ,replace = False)
		k = np.random.choice([1,2,3])
		rot_volume = torch.rot90(volume, int(k), [int(temp[0]), int(temp[1])])
		rot_target = torch.rot90(target, int(k), [int(temp[0]), int(temp[1])])

	return rot_volume, rot_target


def get_random_translation(volume, target, delta = 0.005):
	'''
	Input: tensor
	Output: Randomly translated tensor ?????
	'''
	# rot = tensor.clone()
	
	temp = torch.ones(volume.shape)

	print(temp.shape)


	return volume, target