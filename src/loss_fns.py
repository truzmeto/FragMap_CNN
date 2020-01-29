import torch
import torch.nn as nn


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



    
class PenLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        
    def StepF(self, inp, thresh):
        return 0.5*(1.0 - torch.sign(inp + thresh))
    
    
    def forward(self, output, target, thresh):
        
        loss = torch.abs(output - target) * \
               self.StepF(output + target, -2.0*thresh) - \
               2.0*(output - thresh) * \
               self.StepF(-output - target, 2.0*thresh) * \
               self.StepF(-target, thresh) * \
               self.StepF(output, -thresh) - \
               2.0*(target - thresh) * \
               self.StepF(-output - target, 2.0*thresh) * \
               self.StepF(-output, thresh) * \
               self.StepF( target, -thresh)
        
        
        return loss.mean()



class BinLoss(torch.nn.Module):
    """

    """
    def __init__(self):
        super(BinLoss,self).__init__()


    def BinData(self, inp, Blist):

        batch_size, nchannel, nx, ny, nz = inp.shape
        nvox = nx*ny*nz

        out = torch.empty(batch_size, nchannel, len(Blist))

        for ibatch in range(batch_size):
            for ichan in range(nchannel):

                inp1 = inp[ibatch, ichan,:,:,:].squeeze()
                freqs = torch.tensor([len(inp1[(inp1 >= el[0]) & (inp1 < el[1])]) for el in Blist],
                                     dtype = torch.float32)
                #, requires_grad=True)

                out[ibatch,ichan,:] = freqs / nvox

        return out


    def forward(self, inp, tar, bin_range):

        sl1 = nn.SmoothL1Loss()
        loss1 = sl1(inp,tar)
        loss2 = torch.abs(self.BinData(inp, bin_range)
                          - self.BinData(tar, bin_range)).mean()

        #loss2 = torch.tensor(loss2, requires_grad=True)
        return  0.5*loss1 + 0.5*loss2






