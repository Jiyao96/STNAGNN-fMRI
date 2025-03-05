import numpy as np
import torch 

num_nodes=84
num_snapshots=12
dimension=128

dimX=num_nodes*num_snapshots
dimY=dimension

def positional_encoding_2d():
    temp=10000*torch.ones(1,)
    # create tensor
    encoding=torch.zeros(num_nodes*num_snapshots,dimension)
    for x in range(dimX):
        for y in range(dimY):
            pos=x%num_nodes
            t=x//num_nodes
            if (y%2) == 0:
                encoding[x,y]=torch.sin(pos/torch.pow(temp,y/dimY))*torch.sin((10000+t)/torch.pow(temp,y/dimY))
            else:
                encoding[x,y]=torch.cos(pos/torch.pow(temp,(y-1)/dimY))*torch.cos((10000+t)/torch.pow(temp,(y-1)/dimY))
    return encoding

def positional_encoding_1d():
    temp=10000*torch.ones(1,)
    # create tensor
    encoding=torch.zeros(num_nodes*num_snapshots,dimension)
    for x in range(dimX):
        for y in range(dimY):
            pos=x
            t=x//num_nodes
            if (y%2) == 0:
                encoding[x,y]=torch.sin(pos/torch.pow(temp,y/dimY))
            else:
                encoding[x,y]=torch.cos(pos/torch.pow(temp,(y-1)/dimY))
    return encoding

encoding=positional_encoding_2d()
torch.save(encoding,"pe_128")

# encoding=positional_encoding_1d()
# torch.save(encoding,"pe_128_1d")