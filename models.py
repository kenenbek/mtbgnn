import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv, GCNConv
from torch_geometric.nn import global_mean_pool


class ModelSageConv(torch.nn.Module):
    def __init__(self, rec_features, con_features, n_features):
        super().__init__()
        self.rec_features = rec_features
        self.con_features = con_features
        self.n_features = n_features

        self.init_rec = nn.Linear(rec_features, n_features)
        self.init_con = nn.Linear(con_features, n_features)

        self.conv_rec_to_con_1 = HeteroConv({
            ('reactions', 'to', 'constraints'): init_sage_layer(n_features)
        }, aggr='mean')
        self.conv_con_to_rec_1 = HeteroConv({
            ('constraints', 'to', 'reactions'): init_sage_layer(n_features)
        }, aggr='mean')
        self.conv_rec_to_con_2 = HeteroConv({
            ('reactions', 'to', 'constraints'): init_sage_layer(n_features)
        }, aggr='mean')
        self.conv_con_to_rec_2 = HeteroConv({
            ('constraints', 'to', 'reactions'): init_sage_layer(n_features)
        }, aggr='mean')

        self.output1 = nn.Linear(n_features, n_features)
        self.norm1 = nn.BatchNorm1d(n_features)
        self.output2 = nn.Linear(n_features, n_features)
        self.norm2 = nn.BatchNorm1d(n_features)
        self.output3 = nn.Linear(n_features, 1)


    def forward(self, x_dict, edge_index_dict, batch_mask, y_sign):
        x_dict["reactions"] = self.init_rec(x_dict["reactions"])
        x_dict["constraints"] = self.init_con(x_dict["constraints"])

        output = self.conv_rec_to_con_1(x_dict, edge_index_dict)
        x_dict["constraints"] = output["constraints"]

        output = self.conv_con_to_rec_1(x_dict, edge_index_dict)
        x_dict["reactions"] = output["reactions"]

        output = self.conv_rec_to_con_2(x_dict, edge_index_dict)
        x_dict["constraints"] = output["constraints"]

        output = self.conv_con_to_rec_2(x_dict, edge_index_dict)
        reactions_features = output["reactions"]

        out = self.output1(reactions_features)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.output2(out)
        out = self.norm2(out)
        out = F.relu(out)

        out = global_mean_pool(out, batch_mask)
        out = self.output3(out) * y_sign.unsqueeze(1)
        return out.squeeze(1)


def init_gat_layer(n_features):
    return GATv2Conv(in_channels=n_features,
                     out_channels=n_features,
                     heads=2,
                     concat=False,
                     dropout=0.2,
                     add_self_loops=False,
                     edge_dim=1,
                     )

def init_gcn_layer(n_features):
    return GCNConv(in_channels=n_features,
                   out_channels=n_features,
                   add_self_loops=False
                   )

def init_sage_layer(n_features):
    return SAGEConv(in_channels=n_features,
                    out_channels=n_features,
                    normalize=True,
                    root_weight=True,
                    project=True)

def compute_loss(out, true_fva, S, batch_size):
    fva_loss_min = F.mse_loss(out[:, 0], true_fva[:, 0])
    fva_loss_max = F.mse_loss(out[:, 1], true_fva[:, 1])

    n_reactions = int(S.shape[1])
    n_metabolites = int(S.shape[0] / batch_size)
    S = S.view(batch_size, n_metabolites, n_reactions)

    sv_loss_min = torch.square(torch.matmul(S, out[:, 0].view(batch_size, -1, 1))).mean()
    sv_loss_max = torch.square(torch.matmul(S, out[:, 1].view(batch_size, -1, 1))).mean()

    loss = fva_loss_min + fva_loss_max# + sv_loss_min + sv_loss_max

    return loss