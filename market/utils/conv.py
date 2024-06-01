import torch
from torch import nn
from torch.nn import Conv1d

#TODO: is this file even useful?
class ConvLayers(nn.Module):
    def __init__(self):
        super(ConvLayers, self).__init__()
        # args given to Conv1d are (in_channels, out_channels, kernel size)
        #in_channels of layer0 = out_channels of layer n = 1
        #in_channels of layer k = out_channels of layer (k-1)
        self.conv1 = Conv1d(1,5,5,stride=2)
        self.conv2 = Conv1d(5,3,3,stride=2)
        self.conv3 = Conv1d(3,1,1,stride=1)


    def forward(self,X):
        X = torch.from_numpy(X).float()
        if X.dim() == 1:
            #reshape (m,) to (1,1,m)
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.dim() == 2:
            #reshape (n,m) to (n,1,m)
            X = X.unsqueeze(1)

        out = self.conv1(X)
        out = self.conv2(out)
        out = self.conv3(out)

        return out
        

    #use it just like ConvLayers()(data)