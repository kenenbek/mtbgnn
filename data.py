import os
from pyexpat import features

import torch
from torch_geometric.data import Dataset, HeteroData, InMemoryDataset
from sklearn.metrics import r2_score


class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.file_list = [f for f in os.listdir(root) if f.endswith('.pth') and f not in ["S.pth", "b.pth", "c.pth"]]

    @property
    def raw_file_names(self):
        return self.file_list

    @property
    def processed_file_names(self):
        return self.file_list

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        # Load a graph from a file
        data_path = os.path.join(self.root, self.file_list[idx])
        data = torch.load(data_path, weights_only=False)
        return data


class FilteredDataset(InMemoryDataset):
    def __init__(self, root, data_list):
        self.data, self.slices = self.collate(data_list)
        super(FilteredDataset, self).__init__(root)





class GraphPermutedDataset(Dataset):
    def __init__(self, hetero_data, len_dataset, y_sign):
        """
        Args:
            reactions_features: has to be 2D tensor
            three possible data shape is possible:  N, N, 2N
        """
        super().__init__()
        self.hetero_data = hetero_data
        self.reactions_features_fba = hetero_data["reactions"].x.clone()
        self.reactions_features_fva = hetero_data["reactions"].x.clone()

        self.fva = hetero_data["reactions"].fva

        self.constraints = hetero_data["constraints"]
        self.c_to_r = hetero_data[("constraints", "to", "reactions")]
        self.r_to_c = hetero_data[("reactions", "to", "constraints")]

        self.biomass_index = hetero_data["biomass_index"]
        self.objective_value = hetero_data["objective_value"]
        self.num_reactions = hetero_data["reactions"].x.size(0)

        self.len_dataset = len_dataset
        self.y_sign = y_sign
        self.y_ind = {1:0, -1:1}

    def set_biomass_constraint(self, objective_value):
        self.reactions_features_fva[self.biomass_index] = torch.tensor(
                [objective_value.item(), objective_value.item()], dtype=torch.float64)

    def len(self):
        """Returns the total number of samples."""
        return self.len_dataset

    def get(self, idx):
        y_sign = self.y_sign
        if idx == self.biomass_index or idx == 2 * self.biomass_index:
            features = self.reactions_features_fba
        else:
            features = self.reactions_features_fva

        if idx < self.num_reactions and self.len_dataset == 2 * self.num_reactions:
            idx = idx
            y_sign = 1
        elif idx >= self.num_reactions and self.len_dataset == 2 * self.num_reactions:
            idx = idx - self.num_reactions
            y_sign = -1

        c_vector = torch.zeros(self.num_reactions)
        c_vector[idx] = y_sign
        y = self.fva[idx, self.y_ind[y_sign]]

        features = torch.cat((c_vector.unsqueeze(1), features), dim=1)

        new_graph = HeteroData()

        new_graph["reactions"].x = features
        new_graph["reactions"].y = y
        new_graph["reactions"].y_sign = y_sign

        new_graph["constraints"].x = self.constraints.x
        new_graph[("constraints", "to", "reactions")].edge_index = self.c_to_r.edge_index
        new_graph[("constraints", "to", "reactions")].edge_attr = self.c_to_r.edge_attr

        new_graph[("reactions", "to", "constraints")].edge_index = self.r_to_c.edge_index
        new_graph[("reactions", "to", "constraints")].edge_attr = self.r_to_c.edge_attr
        return new_graph


def create_x_and_edges(hetero_batch, device, fba=False):
    x_dict = {node_type: hetero_batch[node_type].x.to(device) for node_type in hetero_batch.node_types}
    edge_index_dict = {k: hetero_batch[k].edge_index.to(device) for k in hetero_batch.edge_types}
    batch_indices = hetero_batch["reactions"].batch.to(device)
    if not fba:
        y = hetero_batch["reactions"].y.to(device)
        y_sign = hetero_batch["reactions"].y_sign.to(device)
    else:
        index = hetero_batch["biomass_index"].item()
        y = hetero_batch["reactions"].fva[index][1].to(device)
        y_sign = torch.tensor([-1.], dtype=torch.float64).to(device)

        c_vector = torch.zeros(hetero_batch["reactions"].x.size(0)).to(device)
        c_vector[index] = y_sign
        x_dict["reactions"] = torch.cat((c_vector.unsqueeze(1), x_dict["reactions"]), dim=1)

    return x_dict, edge_index_dict, batch_indices, y, y_sign


if __name__ == '__main__':
    # Initialize your dataset
    dataset = GraphDataset(root='my_archive')

    print(dataset)
    x = HeteroData(dataset[0])
    print(x)

    # # Optionally split the dataset into train and test sets
    # train_size = int(len(dataset) * 0.8)
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    #
    # # Create DataLoader for training
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #
    # # Create DataLoader for testing
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)