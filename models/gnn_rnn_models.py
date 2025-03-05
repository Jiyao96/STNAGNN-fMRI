# import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import static layers
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GATConv, TransformerConv, GINConv
# import pooling layers
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

silu=nn.SiLU()

class StaticGNNRNN(nn.Module):
    def __init__(self, conv_type, temporal_type, device):
        super(StaticGNNRNN, self).__init__()
        # record layer types
        self.c_type=conv_type
        self.t_type=temporal_type
        self.device=device

        # graph layers
        if conv_type=="GCN":
            self.conv1=GCNConv(84,128)
            self.conv2=GCNConv(128,128)
        elif conv_type=="Cheb":
            self.conv1=ChebConv(84,128,K=4)
            self.conv2=ChebConv(128,128,K=4)
        elif conv_type=="SAGE":
            self.conv1=SAGEConv(84,128)
            self.conv2=SAGEConv(128,128)
        elif conv_type=="GAT":
            self.conv1=GATConv(84,128,heads=1,edge_dim=1)
            self.conv2=GATConv(128,128,heads=1,edge_dim=1)
        elif conv_type=="Transformer":
            self.conv1=TransformerConv(84,128,heads=1)
            self.conv2=TransformerConv(128,128,heads=1)
        elif conv_type=="GIN":
            self.conv1=GINConv(nn.Linear(84,128))
            self.conv2=GINConv(nn.Linear(128,128))
        else:
            print("Linear layer used in graph.")
            self.conv1=nn.Linear(84,128)
            self.conv2=nn.Linear(128,128)

        # temporal layers
        if temporal_type=="LSTM":
            self.temporal=nn.LSTM(input_size=256,hidden_size=100,num_layers=2)
        elif temporal_type=="biLSTM":
            self.temporal=nn.LSTM(input_size=256,hidden_size=100,num_layers=2,bidirectional=True)
        else:
            print("Temporal method not specified")

        # dropout layer
        self.dp=nn.Dropout(p=0.2)

        # batch normalizations
        self.bn1=nn.BatchNorm1d(512)
        self.bn2=nn.BatchNorm1d(32)

        # Linear layers
        if temporal_type=="biLSTM" or temporal_type=="biGRU":
            self.fc1=nn.Linear(2400,512)
        else:
            self.fc1=nn.Linear(1200,512)
        self.fc2=nn.Linear(512,32)
        self.fc3=nn.Linear(32,2)
    
    def forward(self, data):
        T=len(data)
        x_list=[]
        y_list=np.array(data[0].y)
        y_list=torch.tensor(y_list)
        for i in range(T):
            temp=data[i]
            temp_x0=torch.tensor(temp.x).reshape(-1,84).to(self.device)
            edge_index=temp.edge_index.to(self.device)
            edge_attr=temp.edge_attr.to(self.device)
            temp.batch=temp.batch.to(self.device)
            if self.c_type=="SAGE":
                temp_x1=self.conv1(x=temp_x0,edge_index=edge_index)
                temp_x2=self.conv2(x=temp_x1,edge_index=edge_index)
            elif self.c_type=="GAT": 
                temp_x1=self.conv1(x=temp_x0,edge_index=edge_index,edge_attr=edge_attr)
                temp_x2=self.conv2(x=temp_x1,edge_index=edge_index,edge_attr=edge_attr)
            elif self.c_type=="Transformer" or self.c_type=="GIN":
                temp_x1=self.conv1(x=temp_x0,edge_index=edge_index)
                temp_x2=self.conv2(x=temp_x1,edge_index=edge_index)
            else:
                temp_x1=self.conv1(x=temp_x0,edge_index=edge_index,edge_weight=edge_attr)
                temp_x2=self.conv2(x=temp_x1,edge_index=edge_index,edge_weight=edge_attr)
            aggr_x = torch.cat([gmp(temp_x1, temp.batch), gap(temp_x2, temp.batch)], dim=1)
            x_list.append(aggr_x)
        x_batch=torch.stack(x_list)
        if self.t_type=="LSTM" or self.t_type=="biLSTM":
            out, (h, c)=self.temporal(x_batch)
        if self.t_type=="biLSTM":
            out=torch.swapaxes(out,0,1).reshape(-1,2400)
        else:
            out=torch.swapaxes(out,0,1).reshape(-1,1200)
        out=silu(self.fc1(out))
        out=self.bn1(out)
        out=self.dp(out)
        out=silu(self.fc2(out))
        out=self.bn2(out)
        out=self.dp(out)
        out=F.softmax(self.fc3(out),dim=1)
        return out, y_list