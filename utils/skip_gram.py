import torch
import torch.nn as nn
import numpy as np

GPU = 3
device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)  

    def forward(self, x):
        return self.fc(x)  
        ".detach().view(-1).cpu().numpy()"


class Decoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)  

    def forward(self, h):
        return self.fc(h) 
        ".detach().view(-1).cpu().numpy()"


if __name__ == "__main__":
    pass