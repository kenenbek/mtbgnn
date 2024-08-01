import os
import torch
from torch_geometric.data import Dataset, HeteroData


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


from torch_geometric.loader import DataLoader



# # Example training loop
# for data in train_loader:  # Iterate over each batch
#     out = model(data.x, data.edge_index, data.batch)  # Assuming model is already defined
#     # Compute loss and backpropagate
#     # optimizer.zero_grad()
#     # loss.backward()
#     # optimizer.step()



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