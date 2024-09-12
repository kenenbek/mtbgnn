import os
import torch
from torch_geometric.data import Dataset, HeteroData, InMemoryDataset


class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.file_list = [f for f in os.listdir(root) if f.endswith('.pth')]

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


class FvaInstantDataset(Dataset):
    def __init__(self, reactions_features, constraints, edge_index_dict, edge_features, y, device):
        """
        Args:
            reactions_features: has to be 2D tensor
        """
        super().__init__()
        assert reactions_features.ndimension() == 2, "reactions_features must be a 2D tensor."
        assert reactions_features.size(1) == 2, "reactions_features must have exactly 2 columns."

        self.reactions_features = reactions_features
        self.constraints = constraints
        self.edge_index_dict = edge_index_dict
        self.edge_features = edge_features
        self.y = y
        self.num_reactions = reactions_features.size(0)
        self.device = device

    def len(self):
        """Returns the total number of samples."""
        return 2 * self.num_reactions

    def get(self, idx):
        c_vector = torch.zeros(self.num_reactions)

        if idx < self.num_reactions:
            c_vector[idx] = 1
            label = self.y[idx, 0]
        else:
            c_vector[idx - self.num_reactions] = -1
            label = self.y[idx - self.num_reactions, 1]

        features = torch.cat((c_vector.unsqueeze(1).to(self.device), self.reactions_features), dim=1)

        new_graph = HeteroData()

        new_graph["reactions"].x = features
        new_graph["reactions"].y = label

        new_graph["constraints"].x = self.constraints
        new_graph[("constraints", "to", "reactions")].edge_index = self.edge_index_dict[
            ('constraints', 'to', 'reactions')]
        new_graph[("constraints", "to", "reactions")].edge_attr = self.edge_features

        new_graph[("reactions", "to", "constraints")].edge_index = self.edge_index_dict[
            ('reactions', 'to', 'constraints')]
        new_graph[("reactions", "to", "constraints")].edge_attr = self.edge_features

        return new_graph


def create_x_and_edges(hetero_batch, device):
    x_dict = {node_type: hetero_batch[node_type].x.to(device) for node_type in hetero_batch.node_types}
    edge_index_dict = {k: hetero_batch[k].edge_index.to(device) for k in hetero_batch.edge_types}
    return x_dict, edge_index_dict

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