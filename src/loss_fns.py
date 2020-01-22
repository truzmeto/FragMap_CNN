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

    def forward(self, output, target, thresh):

        l1 = torch.abs(output - target)

        step1 = 0.5*(1.0 - torch.sign( output+ target- 2*thresh))
        step2 = 0.5*(1.0 - torch.sign(-target+ thresh))
        step3 = 0.5*(1.0 - torch.sign( output- thresh))

        step1p  = 0.5*(1.0 - torch.sign(-output- target+ 2*thresh))
        step2p = 0.5*(1.0 - torch.sign(-output+ thresh))
        step3p = 0.5*(1.0 - torch.sign( target- thresh))

        z1 = l1*step1
        z2 = -2*(output - thresh)*step1p*step2 *step3
        z3 = -2*(target - thresh)*step1p*step2p*step3p
        
        loss = z1 + z2 + z3

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






