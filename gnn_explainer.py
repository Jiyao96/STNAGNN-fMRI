from typing import Optional

from math import sqrt
from inspect import signature

import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx
import copy

EPS = 1e-15
temp_batch_size=50
temp_step_size=12
temp_node_num=84
temp_edge_num=354
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNExplainer(torch.nn.Module):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note::

        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        num_hops (int, optional): The number of hops the :obj:`model` is
            aggregating information from.
            If set to :obj:`None`, will automatically try to detect this
            information based on the number of
            :class:`~torch_geometric.nn.conv.message_passing.MessagePassing`
            layers inside :obj:`model`. (default: :obj:`None`)
        return_type (str, optional): Denotes the type of output from
            :obj:`model`. Valid inputs are :obj:`"log_prob"` (the model
            returns the logarithm of probabilities), :obj:`"prob"` (the
            model returns probabilities), :obj:`"raw"` (the model returns raw
            scores) and :obj:`"regression"` (the model returns scalars).
            (default: :obj:`"log_prob"`)
        feat_mask_type (str, optional): Denotes the type of feature mask
            that will be learned. Valid inputs are :obj:`"feature"` (a single
            feature-level mask for all nodes), :obj:`"individual_feature"`
            (individual feature-level masks for each node), and :obj:`"scalar"`
            (scalar mask for each each node). (default: :obj:`"feature"`)
        allow_edge_mask (boolean, optional): If set to :obj:`False`, the edge
            mask will not be optimized. (default: :obj:`True`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in :attr:`~torch_geometric.nn.models.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 1.0, # 0.005,
        'edge_reduction': 'mean', #'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.5, # 1.0
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs: int = 100, lr: float = 0.01,
                 num_hops: Optional[int] = None, return_type: str = 'log_prob',
                 feat_mask_type: str = 'feature', edge_mask_type: str='individual',
                 allow_edge_mask: bool = True,
                 log: bool = True, **kwargs):
        super().__init__()
        assert return_type in ['log_prob', 'prob', 'raw', 'regression']
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar', 'spatial', 'temporal','spatiotemporal']
        assert edge_mask_type in ['scalar', 'spatial','spatiotemporal']
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.return_type = return_type
        self.log = log
        self.allow_edge_mask = allow_edge_mask
        self.feat_mask_type = feat_mask_type
        self.edge_mask_type = edge_mask_type
        self.coeffs.update(kwargs)

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        if self.feat_mask_type == 'individual_feature':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, F) * std)
        elif self.feat_mask_type == 'scalar':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, 1) * std)
        elif self.feat_mask_type == 'spatial':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(temp_node_num,1) * std)
        elif self.feat_mask_type == 'temporal':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(temp_step_size, 1) * std)
        elif self.feat_mask_type == 'spatiotemporal':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(temp_node_num*temp_step_size, 1) * std)
        else:
            self.node_feat_mask = torch.nn.Parameter(torch.randn(1, F) * std)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        if self.edge_mask_type=='scalar':
            self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        elif self.edge_mask_type=='spatial':
            self.edge_mask = torch.nn.Parameter(torch.randn(temp_edge_num) * std)
        else:
            self.edge_mask = torch.nn.Parameter(torch.randn(temp_edge_num * temp_step_size) * std)
        
        if not self.allow_edge_mask:
            self.edge_mask.requires_grad_(False)
            self.edge_mask.fill_(float('inf'))  # `sigmoid()` returns `1`.
        self.loop_mask = edge_index[0] != edge_index[1]

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                if self.edge_mask_type=='scalar':
                    module.__edge_mask__ = self.edge_mask
                elif self.edge_mask_type=='spatial':
                    module.__edge_mask__ = self.edge_mask.repeat(temp_step_size*temp_batch_size)
                else:
                    module.__edge_mask__ = self.edge_mask.repeat(temp_batch_size)
                module.__loop_mask__ = self.loop_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
                module.__loop_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None
        module.loop_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, subset, kwargs

    def __loss__(self, node_idx, log_logits, pred_label):
        # node_idx is -1 for explaining graphs
        if self.return_type == 'regression':
            if node_idx != -1:
                loss = torch.cdist(log_logits[node_idx], pred_label[node_idx])
            else:
                loss = torch.cdist(log_logits, pred_label)
        else:
            if node_idx != -1:
                loss = -log_logits[node_idx, pred_label[node_idx]]
            else:
                loss = -log_logits[0, pred_label[0]]

        m = self.edge_mask.sigmoid()
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        edge_loss=loss
        # print(edge_loss.item())

        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()
        # node_loss=loss-edge_loss
        # print(node_loss.item())
        # print(loss.item())

        return loss

    def __to_log_prob__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.log_softmax(dim=-1) if self.return_type == 'raw' else x
        x = x.log() if self.return_type == 'prob' else x
        return x

    def explain_graph(self, data, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self.__clear_masks__()

        x=data.x
        edge_index=data.edge_index
        edge_attr=data.edge_attr

        # print(edge_attr.shape)

        # Get the initial prediction.
        with torch.no_grad():
            out = self.model(data)
            if self.return_type == 'regression':
                prediction = out
            else:
                log_logits = self.__to_log_prob__(out)
                pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)
        if self.allow_edge_mask:
            parameters = [self.node_feat_mask, self.edge_mask]
        else:
            parameters = [self.node_feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description('Explain graph')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()

            # apply node masks
            if self.feat_mask_type=="spatial":
                # spatial importance of nodes
                node_temp_mask = self.node_feat_mask.sigmoid().repeat(temp_batch_size*temp_step_size,1)
            elif self.feat_mask_type == "temporal":
                # temporal importance of graphs
                node_temp_mask = self.node_feat_mask.sigmoid().repeat_interleave(temp_node_num*temp_batch_size,0)
            elif self.feat_mask_type == "spatiotemporal":
                node_temp_mask = self.node_feat_mask.sigmoid().repeat(temp_batch_size,1)
            else:
                node_temp_mask = self.node_feat_mask.sigmoid()
            h = x * node_temp_mask

            # apply edge masks
            if self.edge_mask_type=="scalar":
                edge_temp_mask = self.edge_mask.sigmoid()
            elif self.edge_mask_type=="spatial":
                edge_temp_mask = self.edge_mask.sigmoid().repeat(temp_batch_size*temp_step_size)
            else:
                edge_temp_mask = self.edge_mask.sigmoid().repeat(temp_batch_size)

            g = edge_attr * edge_temp_mask
            mask_data=copy.deepcopy(data)
            mask_data.x=h
            mask_data.edge_attr=g

            # feed masked data to trained model
            out = self.model(mask_data)
            if self.return_type == 'regression':
                loss = self.__loss__(-1, out, prediction)
            else:
                log_logits = self.__to_log_prob__(out)
                loss = self.__loss__(-1, log_logits, pred_label)
            loss.backward()
            optimizer.step()

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid().squeeze()
        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()
        return node_feat_mask, edge_mask

    def __repr__(self):
        return f'{self.__class__.__name__}()'
