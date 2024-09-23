from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import numpy as np

from data import GraphDataset, GraphPermutedDataset, create_x_and_edges
from torch.utils.data import ConcatDataset
from models import ModelSageConv, compute_loss
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import argparse
import torch.multiprocessing as mp


def fva_pass_without_grad(model, single_hetero_graph, len_dataset, obj_value, y_sign, batch_size, num_workers, device):
    dataset = GraphPermutedDataset(single_hetero_graph,
                                          len_dataset=len_dataset,
                                           y_sign=y_sign
                                  )
    dataset.set_biomass_constraint(obj_value)

    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            follow_batch=["reactions", "constraints"]
                            )

    outs = []
    true_y = []

    for batch in val_loader:
        x_dict, edge_index_dict, batch_indices, y, y_sign = create_x_and_edges(batch, device)
        out = model(x_dict, edge_index_dict, batch_indices, y_sign)
        outs.extend(out.cpu().tolist())
        true_y.extend(y.cpu().tolist())

    r2 = r2_score(true_y, outs)
    return r2

if __name__ == '__main__':
    experiment = Experiment(
        project_name="tb",
        workspace="kenenbek"
    )
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="An argument parser")
    parser.add_argument('--data', type=str, help='Path to the data file', default="small_test_run")
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=10)
    parser.add_argument('--features', type=int, help='number of features', default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    batch_size = 64
    n_epochs = args.epochs
    n_features = args.features
    num_workers = 4

    model = ModelSageConv(rec_features=3, con_features=1, n_features=n_features).double()
    model.to(device)

    dataset = GraphDataset(root=args.data)

    train_size = int(len(dataset) * 0.7)
    valid_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                               [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              follow_batch=["reactions", "constraints"])
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                              follow_batch=["reactions", "constraints"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             follow_batch=["reactions", "constraints"])

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    with experiment.train():
        step = 0
        for epoch in range(n_epochs):
            experiment.log_current_epoch(epoch)
            model.train()

            for single_hetero_graph in train_loader:
                len_dataset = 2 * single_hetero_graph["reactions"].x.shape[0]
                permutation_dataset = GraphPermutedDataset(single_hetero_graph,
                                                           len_dataset=len_dataset,
                                                           y_sign=None)
                permute_train_loader = DataLoader(permutation_dataset,
                                                  batch_size=64,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  follow_batch=["reactions", "constraints"]
                                                  )
                for permute_batch in permute_train_loader:
                    optimizer.zero_grad()
                    x_dict, edge_index_dict, batch_indices, y, y_sign = create_x_and_edges(permute_batch, device)
                    out = model(x_dict, edge_index_dict, batch_indices, y_sign)
                    loss = F.mse_loss(out, y)
                    experiment.log_metric("loss", loss.item(), step=epoch)
                    loss.backward()
                    optimizer.step()

            if epoch % 3 == 0:
                model.eval()
                with torch.no_grad():
                    r2_min_val = []
                    r2_max_val = []
                    for single_hetero_graph in valid_loader:
                        x_dict, edge_index_dict, batch_indices, y, y_sign = create_x_and_edges(single_hetero_graph,
                                                                                               device,
                                                                                               fba=True)
                        obj_value = model(x_dict, edge_index_dict, batch_indices, y_sign)
                        len_dataset = single_hetero_graph["reactions"].x.shape[0]

                        r2_min = fva_pass_without_grad(model=model,
                                                       single_hetero_graph=single_hetero_graph,
                                                       len_dataset=len_dataset,
                                                       obj_value=obj_value,
                                                       y_sign=1,
                                                       batch_size=batch_size,
                                                       num_workers=num_workers,
                                                       device=device)

                        r2_max = fva_pass_without_grad(model=model,
                                                       single_hetero_graph=single_hetero_graph,
                                                       len_dataset=len_dataset,
                                                       obj_value=obj_value,
                                                       y_sign=-1,
                                                       batch_size=batch_size,
                                                       num_workers=num_workers,
                                                       device=device)
                        r2_min_val.append(r2_min)
                        r2_max_val.append(r2_max)

                    experiment.log_metric("r2_min_val", np.mean(r2_min_val), step=epoch)
                    experiment.log_metric("r2_max_val", np.mean(r2_max_val), step=epoch)

    ## testing
    with torch.no_grad():
        model.eval()
        r2_min_test = []
        r2_max_test = []
        for single_hetero_graph in test_loader:
            x_dict, edge_index_dict, batch_indices, y, y_sign = create_x_and_edges(single_hetero_graph,
                                                                                   device,
                                                                                   fba=True)
            obj_value = model(x_dict, edge_index_dict, batch_indices, y_sign)
            len_dataset = single_hetero_graph["reactions"].x.shape[0]

            r2_min = fva_pass_without_grad(model=model,
                                           single_hetero_graph=single_hetero_graph,
                                           len_dataset=len_dataset,
                                           obj_value=obj_value,
                                           y_sign=1,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           device=device)

            r2_max = fva_pass_without_grad(model=model,
                                           single_hetero_graph=single_hetero_graph,
                                           len_dataset=len_dataset,
                                           obj_value=obj_value,
                                           y_sign=-1,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           device=device)
            r2_min_test.append(r2_min)
            r2_max_test.append(r2_max)
    experiment.log_metric("r2_min_test", np.mean(r2_min_test))
    experiment.log_metric("r2_max_test", np.mean(r2_max_test))

    # Save the model
    torch.save(model.state_dict(), "model.pth")

    # Log the model file to Comet ML
    experiment.log_model("model", "model.pth")
