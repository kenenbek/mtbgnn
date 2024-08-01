import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class ModelSageConv(torch.nn.Module):
    def __init__(self, rec_features, con_features, n_features):
        super().__init__()
        self.rec_features = rec_features
        self.con_features = con_features
        self.n_features = n_features

        self.init_rec = nn.Linear(rec_features, n_features)
        self.init_con = nn.Linear(con_features, n_features)

        self.conv_rec_to_con_1 = HeteroConv({
            ('reactions', 'to', 'constraints'): SAGEConv(n_features, n_features,
                                                         normalize=True,
                                                         root_weight=True,
                                                         project=True)
        }, aggr='mean')
        self.conv_con_to_rec_1 = HeteroConv({
            ('constraints', 'to', 'reactions'): SAGEConv(n_features, n_features,
                                                         normalize=True,
                                                         root_weight=True,
                                                         project=True
                                                         )
        }, aggr='mean')
        self.conv_rec_to_con_2 = HeteroConv({
            ('reactions', 'to', 'constraints'): SAGEConv(n_features, n_features,
                                                         normalize=True,
                                                         root_weight=True,
                                                         project=True
                                                         )
        }, aggr='mean')
        self.conv_con_to_rec_2 = HeteroConv({
            ('constraints', 'to', 'reactions'): SAGEConv(n_features, n_features,
                                                         normalize=True,
                                                         root_weight=True,
                                                         project=True
                                                         )
        }, aggr='mean')

        self.output1 = nn.Linear(n_features, n_features)
        self.output2 = nn.Linear(n_features, 2)

    def forward(self, x_dict, edge_index_dict):
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
        out = F.relu(out)
        out = self.output2(out)

        return out


def compute_loss(out, true_fva, S, batch_size):
    fva_loss_min = F.mse_loss(out[:, 0], true_fva[:, 0])
    fva_loss_max = F.mse_loss(out[:, 1], true_fva[:, 1])

    n_reactions = int(S.shape[1])
    n_metabolites = int(S.shape[0] / batch_size)
    S = S.view(batch_size, n_metabolites, n_reactions)

    sv_loss_min = torch.square(torch.matmul(S, out[:, 0].view(batch_size, -1, 1))).mean()
    sv_loss_max = torch.square(torch.matmul(S, out[:, 1].view(batch_size, -1, 1))).mean()

    loss = fva_loss_min + fva_loss_max + sv_loss_min + sv_loss_max

    return loss