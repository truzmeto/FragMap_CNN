import torch
import torch.nn as nn
import numpy as np

class logCoshLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, output, target):
        diff = output - target
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))

class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        diff = output - target
        return torch.mean(diff * torch.tanh(diff))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        diff = output - target
        return torch.mean(2 * diff / (1 + torch.exp(-diff)) - diff)


    
class BinLoss(torch.nn.Module):
    """Arth """
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


def get_bin_frequency(gfe_map):
    # Return the frequency of GFE values in each binpossibly Variable-torch
    RT = 0.59248368 
    cp1= np.copy(gfe_map)
    cp2= np.copy(gfe_map)
    cp3= np.copy(gfe_map)
    cp4= np.copy(gfe_map)
    cp5= np.copy(gfe_map)

    cp1 = cp1[cp1>0.25*RT]
    cp2 = cp2[cp2>0]
    cp2 = cp2[cp2<0.25*RT]
    cp3 = cp3[cp3<0]
    cp3 = cp3[cp3>-0.25*RT]
    cp4 = cp4[cp4<-0.25*RT]
    cp4 = cp4[cp4>-0.5*RT]
    cp5 = cp5[cp5<-0.5*RT]

    freq_list = [len(cp1), len(cp2), len(cp3), len(cp4) ,len(cp5)]
    return freq_list
