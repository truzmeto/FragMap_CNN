import torch
import torch.nn as nn
import numpy as np
from src.util import get_bin_frequency


class BinLoss(torch.nn.Module):
    
    def __init__(self):
        super(BinLoss,self).__init__()
        
    def forward(self, pred, target):

    	criterion = nn.MSELoss()
    	rot_target = torch.from_numpy(target).float().cuda()
    	loss_1 = criterion(pred,rot_target)

    	pred =  pred.cpu().detach().numpy()
    	loss2 = 0 
    	#  Main loss as  CrossEntropy
    	# loss2 = Average of differences in bin Frequencies 
    	# bin_frequency_measured, bin_frequency_predicted \
    	#np.avg([np.abs(bin_frequency_measured(ii) - bin_frequency_predicted(ii)) for ii in len(bin_frequency_measured)])
    	# weight_set = [0.1,0.1,0.2,0.3,0.3]
    	freq_list_pred = get_bin_frequency(pred)
    	freq_list_orig = get_bin_frequency(target)

    	bin_diff = [(freq_list_orig[i]-freq_list_pred[i]) for i in range(len(freq_list_orig))]
    	scaled_bin_diff = [(bin_diff[i]/np.sum(bin_diff)) for i in range(len(freq_list_orig))]

    	loss2 = np.sum(scaled_bin_diff)
    	final_loss = loss_1 + loss2
    	return final_loss
