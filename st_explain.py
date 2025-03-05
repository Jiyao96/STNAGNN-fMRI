import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from imports.MultigraphData import MultigraphDataset
from torch_geometric.loader import DataLoader
from temporal_models.gnn_att_models import GNN_att
from gnn_explainer import GNNExplainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load trained gnn model
model = GNN_att("GAT", "local", 50)
model.load_state_dict(torch.load("PATH TO TRAINED MODEL"))
model.to(device)

# create explainer object
explainer = GNNExplainer(model, epochs=50, lr=1e-3, feat_mask_type="spatiotemporal", edge_mask_type="spatiotemporal", allow_edge_mask=False)

# load explanation data
paths=torch.load("PATH TO DATASET")
test_dataset=MultigraphDataset(paths,1)
# load entire dataset in 1 batch
test_loader=DataLoader(test_dataset,batch_size=50,shuffle=True)

# explain data

for data in test_loader:
    data=data.to(device)
    f_mask, e_mask=explainer.explain_graph(data)
    print(f_mask.shape)
    print(e_mask.shape)
    torch.save(f_mask,"SAVE DIR1")
    torch.save(e_mask,"SAVE DIR2")
    break


