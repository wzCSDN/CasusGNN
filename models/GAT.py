from enum import Enum, auto
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from models.dp_gat import DotProductGATConv
from models.gat2 import GAT2Conv
# from models.dp_gat_new import DotProductGATConv
from tqdm import tqdm
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import torch.nn as nn

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

class GAT(torch.nn.Module):
    def __init__(self, base_layer, in_channels, hidden_channels, out_channels, num_layers, num_heads,
                 dropout, device, saint, use_layer_norm, use_residual, use_resdiual_linear,use_causal_reg, num_prototypes):
        super(GAT, self).__init__()

        self.layers = torch.nn.ModuleList()
        kwargs = {
            'bias': False
        }
        if base_layer is GAT2Conv:
            kwargs['share_weights'] = True
        self.layers.append(base_layer(in_channels, hidden_channels // num_heads, num_heads, **kwargs))
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.use_resdiual_linear = use_resdiual_linear
        self.layer_norms = torch.nn.ModuleList()
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        self.residuals = torch.nn.ModuleList()
        if use_resdiual_linear and use_residual:
            self.residuals.append(nn.Linear(in_channels, hidden_channels))
        self.num_layers = num_layers
        for _ in range(num_layers - 2):
            self.layers.append(
                base_layer(hidden_channels, hidden_channels // num_heads, num_heads, **kwargs))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
            if use_resdiual_linear and use_residual:
                self.residuals.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(base_layer(hidden_channels, out_channels, 1, **kwargs))
        if use_resdiual_linear and use_residual:
            self.residuals.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout
        self.device = device
        self.saint = saint
        self.non_linearity = F.relu

        self.use_causal_reg = use_causal_reg
        if self.use_causal_reg:
            self.num_prototypes = num_prototypes

            # 1. 纯净因果特征提取 MLP_x (h_x = MLP(X))
            self.mlp_x = Linear(in_channels, hidden_channels)

            # 2. 环境摘要GNN (GNN_env) - 拥有独立的GNN层
            self.env_layers = torch.nn.ModuleList()
            self.env_layers.append(base_layer(in_channels, hidden_channels // num_heads, num_heads, **kwargs))
            for _ in range(num_layers - 1):
                self.env_layers.append(base_layer(hidden_channels, hidden_channels // num_heads, num_heads, **kwargs))

            # 3. 可学习的环境原型库 C
            self.env_prototypes = Parameter(torch.Tensor(num_prototypes, hidden_channels))

            # 4. 干预消息生成器 MLP_msg
            self.msg_mlp = nn.Sequential(
                Linear(hidden_channels + hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.LayerNorm(hidden_channels),
                Linear(hidden_channels, hidden_channels)
            )

            # 5. 因果表示的分类头
            self.causal_predictor = Linear(hidden_channels, out_channels)
        print(f"learnable_params: {sum(p.numel() for p in list(self.parameters()) if p.requires_grad)}")

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.layer_norms:
            layer.reset_parameters()
        for layer in self.residuals:
            layer.reset_parameters()

        if self.use_causal_reg:
            glorot(self.env_prototypes)
            self.mlp_x.reset_parameters()
            for layer in self.env_layers: layer.reset_parameters()
            for layer in self.msg_mlp:
                if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
            self.causal_predictor.reset_parameters()

    def forward_neighbor_sampler(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            new_x = self.layers[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                new_x = self.non_linearity(new_x)
            if 0 < i < self.num_layers - 1 and self.use_residual:
                x = new_x + x_target
            else:
                x = new_x
            if i < self.num_layers - 1:
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def exp_forward_neighbor_sampler(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            new_x = self.layers[i]((x, x_target), edge_index)
            if self.use_residual:
                if self.use_resdiual_linear:
                    x = new_x + self.residuals[i](x_target)
                else:
                    x = new_x + x_target
            else:
                x = new_x

            if i < self.num_layers - 1:
                x = self.non_linearity(x)
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if not self.training:
            return x, None
        return x

    def forward_saint(self, x, adj_t):
        for i, layer in enumerate(self.layers[:-1]):
            new_x = layer(x, adj_t)
            new_x = self.non_linearity(new_x)
            # residual
            if i > 0 and self.use_residual:
                if self.use_resdiual_linear:
                    x = new_x + self.residuals[i](x)
                else:
                    x = new_x + x
                x = new_x + x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj_t)
        return x

    def forward(self, x, adjs):
        if self.saint:
            y_main_logits = self.forward_saint(x, adjs)
        else:
            y_main_logits = self.forward_neighbor_sampler(x, adjs)
        if not self.training or not self.use_causal_reg:
            return y_main_logits, None

        # 1. 确定原始特征输入 x_src
        x_src = x[0] if isinstance(x, tuple) else x
        # 2. 计算纯净特征 h_x
        h_x = self.mlp_x(x_src)

        # 3. 计算环境摘要 h_e (运行独立的 GNN_env)
        h_e = x
        if not self.saint:  # Neighbor Sampler
            for i in range(self.num_layers):
                x_target = h_e[:adjs[i][2][1]]
                h_e = self.env_layers[i]((h_e, x_target), adjs[i][0])
                if i < self.num_layers - 1:
                    h_e = self.non_linearity(h_e)
        else:  # Full graph / SAINT
            for i, layer in enumerate(self.env_layers):
                h_e = layer(h_e, adjs)
                if i < self.num_layers - 1:
                    h_e = self.non_linearity(h_e)

        # 4. 聚类：使用向量内积进行软分配
        h_e_norm = F.normalize(h_e, p=2, dim=1, eps=1e-12)
        prototypes_norm = F.normalize(self.env_prototypes, p=2, dim=1, eps=1e-12)
        scores = h_e_norm @ prototypes_norm.t()
        temperature = 0.1

        # d. Calculate the soft assignment probabilities. This step is now safe from overflow.
        q = F.softmax(scores / temperature, dim=1)

        # 5. 计算边际概率 p (在当前batch的输出节点上计算)
        p = q.mean(dim=0)

        # 6. 从h_x中抽取出与最终输出对应的部分
        num_batch_nodes = y_main_logits.size(0)
        # NeighborLoader保证目标节点在输入x_src的前面
        h_x_batch = h_x[:num_batch_nodes]


        h_x_expanded = h_x_batch.unsqueeze(1).expand(-1, self.num_prototypes, -1)
        prototypes_expanded = self.env_prototypes.unsqueeze(0).expand(num_batch_nodes, -1, -1)
        combined_features = torch.cat([h_x_expanded, prototypes_expanded], dim=-1)
        messages = self.msg_mlp(combined_features)

        p_reshaped = p.view(1, self.num_prototypes, 1)
        h_prime = (messages * p_reshaped).sum(dim=1)


        y_causal_logits = self.causal_predictor(h_prime)

        return y_main_logits, y_causal_logits


    def inference(self, x, subgraph_loader):
        pbar = tqdm(total=x.size(0) * len(self.layers), leave=False, desc="Layer", disable=False)
        pbar.set_description('Evaluating')
        for i, layer in enumerate(self.layers[:-1]):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(self.device)
                x_source = x[n_id].to(self.device)
                x_target = x_source[:size[1]]  # Target nodes are always placed first.
                new_x = layer((x_source, x_target), edge_index)
                new_x = self.non_linearity(new_x)
                # residual
                if i > 0 and self.use_residual:
                    x_target = new_x + x_target
                else:
                    x_target = new_x
                if self.use_layer_norm:
                    x_target = self.layer_norms[i](x_target)
                # x_target = F.dropout(x_target, p=self.dropout, training=self.training)
                xs.append(x_target.cpu())
                pbar.update(batch_size)
            x = torch.cat(xs, dim=0)
        xs = []
        for batch_size, n_id, adj in subgraph_loader:
            edge_index, _, size = adj.to(self.device)
            x_source = x[n_id].to(self.device)
            x_target = x_source[:size[1]]  # Target nodes are always placed first.
            new_x = self.layers[-1]((x_source, x_target), edge_index)
            xs.append(new_x.cpu())
            pbar.update(batch_size)
        x = torch.cat(xs, dim=0)
        pbar.close()
        return x

    def exp_inference(self, x, subgraph_loader):
        pbar = tqdm(total=x.size(0) * len(self.layers), leave=False, desc="Layer", disable=False)
        pbar.set_description('Evaluating')
        for i, layer in enumerate(self.layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(self.device)
                x_source = x[n_id].to(self.device)
                x_target = x_source[:size[1]]  # Target nodes are always placed first.
                new_x = layer((x_source, x_target), edge_index)
                if self.use_residual:
                    if self.use_resdiual_linear:
                        x_target = new_x + self.residuals[i](x_target)
                    else:
                        x_target = new_x + x_target
                else:
                    x_target = new_x
                if i < self.num_layers - 1:
                    x_target = self.non_linearity(x_target)
                    if self.use_layer_norm:
                        x_target = self.layer_norms[i](x_target)

                xs.append(x_target.cpu())
                pbar.update(batch_size)
            x = torch.cat(xs, dim=0)
        pbar.close()
        return x


class GAT_TYPE(Enum):
    GAT = auto()
    DPGAT = auto()
    GAT2 = auto()

    @staticmethod
    def from_string(s):
        try:
            return GAT_TYPE[s]
        except KeyError:
            raise ValueError()

    def __str__(self):
        if self is GAT_TYPE.GAT:
            return "GAT"
        elif self is GAT_TYPE.DPGAT:
            return "DPGAT"
        elif self is GAT_TYPE.GAT2:
            return "GAT2"
        return "NA"

    def get_model(self, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, device, saint,
                  use_layer_norm, use_residual, use_resdiual_linear,use_causal_reg, num_prototypes):
        if self is GAT_TYPE.GAT:
            return GAT(GATConv, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, device,
                       saint, use_layer_norm, use_residual, use_resdiual_linear,use_causal_reg, num_prototypes)
        elif self is GAT_TYPE.DPGAT:
            return GAT(DotProductGATConv, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout,
                       device, saint, use_layer_norm, use_residual, use_resdiual_linear,use_causal_reg, num_prototypes)
        elif self is GAT_TYPE.GAT2:
            return GAT(GAT2Conv, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, device,
                       saint, use_layer_norm, use_residual, use_resdiual_linear,use_causal_reg, num_prototypes)

    def get_base_layer(self):
        if self is GAT_TYPE.GAT:
            return GATConv
        elif self is GAT_TYPE.DPGAT:
            return DotProductGATConv
        elif self is GAT_TYPE.GAT2:
            return GAT2Conv