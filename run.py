from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import numpy as np

from data import GraphDataset
from models import ModelSageConv, compute_loss
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import argparse


if __name__ == '__main__':
    experiment = Experiment(
        project_name="tb",
        workspace="kenenbek"
    )

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

    model = ModelSageConv(rec_features=3, con_features=1, n_features=n_features).double()
    model.to(device)

    dataset = GraphDataset(root=args.data)

    train_size = int(len(dataset) * 0.7)
    valid_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                               [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              follow_batch=["reactions", "constraints"])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              follow_batch=["reactions", "constraints"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             follow_batch=["reactions", "constraints"])

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    with experiment.train():
        step = 0
        for epoch in range(n_epochs):
            experiment.log_current_epoch(epoch)
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                c_batch_size = int((torch.max(batch["reactions"].batch) + 1).item())
                x_dict = {k: batch[k].x.to(device) for k in batch.node_types}
                edge_index_dict = {k: batch[k].edge_index.to(device) for k in batch.edge_types}

                out = model(x_dict, edge_index_dict)
                loss = compute_loss(out, batch["reactions"]["y"].to(device), batch["S"].to(device), c_batch_size)
                total_loss += loss.item()

                r2_min = r2_score(batch["reactions"]["y"][:, 0], out[:, 0].cpu().detach().numpy())
                r2_max = r2_score(batch["reactions"]["y"][:, 1], out[:, 1].cpu().detach().numpy())

                experiment.log_metric("loss_step", loss.item(), epoch=step)
                experiment.log_metric("r2_min", r2_min, epoch=step)
                experiment.log_metric("r2_max", r2_max, epoch=step)
                step += 1

                loss.backward()
                optimizer.step()

            average_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {average_loss:.4f}')
            experiment.log_metric("loss_epoch", average_loss, epoch=epoch)

            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss = 0
                    for batch in valid_loader:

                        c_batch_size = int((torch.max(batch["reactions"].batch) + 1).item())
                        x_dict = {k: batch[k].x.to(device) for k in batch.node_types}
                        edge_index_dict = {k: batch[k].edge_index.to(device) for k in batch.edge_types}

                        out = model(x_dict,
                                    edge_index_dict)
                        val_loss = compute_loss(out, batch["reactions"]["y"].to(device), batch["S"].to(device), c_batch_size)
                        validation_loss += val_loss.item()

                        r2_min = r2_score(batch["reactions"]["y"][:, 0], out[:, 0].cpu().detach().numpy())
                        r2_max = r2_score(batch["reactions"]["y"][:, 1], out[:, 1].cpu().detach().numpy())

                    validation_loss /= len(valid_loader)
                    print(f"Epoch {epoch + 1}, Validation Loss: {validation_loss:.4f}")
                    experiment.log_metric("loss_val", validation_loss)

    r2_min_test = []
    r2_max_test = []
    with torch.no_grad():
        for data in test_loader:
            x_dict = {k: batch[k].x.to(device) for k in batch.node_types}
            edge_index_dict = {k: batch[k].edge_index.to(device) for k in batch.edge_types}

            out = model(x_dict,
                        edge_index_dict)

            r2_min = r2_score(batch["reactions"]["y"][:, 0], out[:, 0].cpu().detach().numpy())
            r2_max = r2_score(batch["reactions"]["y"][:, 1], out[:, 1].cpu().detach().numpy())

            r2_min_test.append(r2_min)
            r2_max_test.append(r2_max)

    print("r2_min_test: ", np.mean(r2_min_test))
    print("r2_max_test: ", np.mean(r2_max_test))

    experiment.log_metric("r2_min_test", np.mean(r2_min_test))
    experiment.log_metric("r2_max_test", np.mean(r2_max_test))

    # Save the model
    torch.save(model.state_dict(), "model.pth")

    # Log the model file to Comet ML
    experiment.log_model("model", "model.pth")
