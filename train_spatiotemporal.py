import h5py
import os
import numpy as np
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
import torch
import torch.nn as nn
from imports.MultigraphData import MultigraphDataset
from torch_geometric.loader import DataLoader
from models.gnn_att_models import GNN_att
# import scheduler
from torch.optim import lr_scheduler
# import evaluation metrics
from sklearn.metrics import f1_score, auc, roc_curve
import copy


# parameters
data_dir="TRAIN DATA DIR"
test_dir="TEST DATA DIR"
batch_size=10
learning_rate=1e-5
weight_decay=0.01
conv_type="GAT"
aggr_type="local"
opt_method="SGD"
num_epoch=50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# read files from directory and sort filenames
paths=torch.load(data_dir+"filenames")
train_dataset=MultigraphDataset(paths,1)
paths=torch.load(test_dir+"filenames")
test_dataset=MultigraphDataset(paths,1)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# define model
model=GNN_att(conv_type, aggr_type, batch_size)
model=model.to(device)
# define optimization
if opt_method=="Adam":
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.01)
elif opt_method=="SGD":
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=1,weight_decay=0.01,nesterov=True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4) 

ce=nn.CrossEntropyLoss()

def train():
    model.train()
    loss_all=0
    correct=0
    for data in train_loader:
        optimizer.zero_grad()
        data=data.to(device)
        out=model(data)
        loss=ce(out,data.y.long())
        loss.backward()
        loss_all+=loss.item()*batch_size
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y.long()).sum().item()
        optimizer.step()
    scheduler.step()
    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    return loss_all/len(train_dataset), correct/len(train_dataset)*100

def test():
    model.eval()
    correct=0
    out_list=[]
    pred_list=[]
    y_list=[]
    loss_all=0
    for data in test_loader:
        data=data.to(device)
        out=model(data)
        loss_all+=ce(out,data.y.long()).item()*batch_size
        pred = out.max(dim=1)[1]
        out_list.append(out)
        pred_list.append(pred)
        y_list.append(data.y.long())
        correct += pred.eq(data.y.long()).sum().item()
    out_all=torch.stack(out_list).reshape(-1,2).cpu()
    pred_all=torch.stack(pred_list).reshape(-1).cpu()
    y_all=torch.stack(y_list).reshape(-1).cpu()
    m1=f1_score(y_all, pred_all, zero_division=1.0)
    fpr, tpr, thresholds = roc_curve(y_all, pred_all, pos_label=1)
    m2=auc(fpr, tpr)
    return loss_all/len(test_dataset), correct/len(test_dataset)*100, m1, m2

for i in range(num_epoch):
    train_loss, train_acc=train()
    test_loss, test_acc, m1, m2=test()
    print("Epoch: ", i+1)
    print("train loss: "+str(train_loss)+" train acc: "+str(train_acc))
    print("test loss: "+str(test_loss)+" test acc: "+str(test_acc)+" f1_score: "+str(m1)+" auc: "+str(m2))









