import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from gcn import *
from logger import Logger

lr = 0.01
epochs = 300
eval_steps = 10
num_layers = 3
dropout = 0.5
log_steps = 10
hidden_channels = 256

device = "cpu"
device = torch.device(device)

dataset_protein = PygNodePropPredDataset(name="ogbn-proteins")
split_idx_protein = dataset_protein.get_idx_split()
data_protein = dataset_protein[0]

x_protein = scatter(
    data_protein.edge_attr,
    data_protein.edge_index[0],
    dim=0,
    dim_size=data_protein.num_nodes,
    reduce="mean",
).to("cpu")

x_protein = x_protein.to(device)
y_protein_true = data_protein.y.to(device)
train_idx_protein = split_idx_protein["train"].to(device)

dataset_product = PygNodePropPredDataset(name="ogbn-products")
split_idx_product = dataset_product.get_idx_split()
data_product = dataset_product[0]

x_product = scatter(
    data_product.edge_attr,
    data_product.edge_index[0],
    dim=0,
    dim_size=data_product.num_nodes,
    reduce="mean",
).to("cpu")

x_product = x_product.to(device)
y_product_true = data_product.y.to(device)
train_idx_product = split_idx_product["train"].to(device)


def train_product():
    model = GCN(
        data_product.num_features, hidden_channels, 112, num_layers, dropout
    ).to(device)

    # Pre-compute GCN normalization.
    adj_t = data_product.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data_product.adj_t = adj_t

    data = data.to(device)

    evaluator = Evaluator(name="ogbn-products")
    logger = Logger(3)

    for run in range(3):
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, 1 + epochs):
            loss = train(model, data, train_idx_product, optimizer)

            if epoch % eval_steps == 0:
                result = test(model, data, split_idx_product, evaluator)
                logger.add_result(run, result)

                if epoch % log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(
                        f"Run: {run + 1:02d}, "
                        f"Epoch: {epoch:02d}, "
                        f"Loss: {loss:.4f}, "
                        f"Train: {100 * train_rocauc:.2f}%, "
                        f"Valid: {100 * valid_rocauc:.2f}% "
                        f"Test: {100 * test_rocauc:.2f}%"
                    )
                    with open("log_gnn2.txt", "a") as f:
                        f.write(
                            f"Run: {run + 1:02d}, "
                            f"Epoch: {epoch:02d}, "
                            f"Loss: {loss:.4f}, "
                            f"Train: {100 * train_rocauc:.2f}%, "
                            f"Valid: {100 * valid_rocauc:.2f}% "
                            f"Test: {100 * test_rocauc:.2f}%\n"
                        )

        logger.print_statistics(run)

        # save model
        torch.save(model.state_dict(), "gnn_proteins.pt")

    logger.print_statistics()


def train_protein():
    model = GCN(
        data_protein.num_features, hidden_channels, 112, num_layers, dropout
    ).to(device)

    # Pre-compute GCN normalization.
    adj_t = data_protein.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data_protein.adj_t = adj_t

    data = data.to(device)

    evaluator = Evaluator(name="ogbn-proteins")
    logger = Logger(3)

    for run in range(3):
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, 1 + epochs):
            loss = train(model, data, train_idx_protein, optimizer)

            if epoch % eval_steps == 0:
                result = test(model, data, split_idx_protein, evaluator)
                logger.add_result(run, result)

                if epoch % log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(
                        f"Run: {run + 1:02d}, "
                        f"Epoch: {epoch:02d}, "
                        f"Loss: {loss:.4f}, "
                        f"Train: {100 * train_rocauc:.2f}%, "
                        f"Valid: {100 * valid_rocauc:.2f}% "
                        f"Test: {100 * test_rocauc:.2f}%"
                    )
                    with open("log_gnn2.txt", "a") as f:
                        f.write(
                            f"Run: {run + 1:02d}, "
                            f"Epoch: {epoch:02d}, "
                            f"Loss: {loss:.4f}, "
                            f"Train: {100 * train_rocauc:.2f}%, "
                            f"Valid: {100 * valid_rocauc:.2f}% "
                            f"Test: {100 * test_rocauc:.2f}%\n"
                        )

        logger.print_statistics(run)

        # save model
        torch.save(model.state_dict(), "gnn_proteins.pt")


def transfer_protein_to_product():
    model = GCN(
        data_product.num_features, hidden_channels, 112, num_layers, dropout
    ).to(device)

    adj_t = data.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t

    pretrained_model = GCN(100, hidden_channels, 47, num_layers, dropout)

    pretrained_model.load_state_dict(torch.load("gnn_protein.pt"))

    for param, pretrained_param in zip(
        model.convs[1].parameters(), pretrained_model.convs[1].parameters()
    ):
        param.data = pretrained_param.data.clone()  # Clone the parameter data
        param.requires_grad = False  # Freeze the parameter

    data = data.to(device)

    evaluator = Evaluator(name="ogbn-proteins")
    logger = Logger(3)

    for run in range(3):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, 1 + epochs):
            loss = train(model, data, train_idx_product, optimizer)

            if epoch % eval_steps == 0:
                result = test(model, data, split_idx_product, evaluator)
                logger.add_result(run, result)

                if epoch % log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(
                        f"Run: {run + 1:02d}, "
                        f"Epoch: {epoch:02d}, "
                        f"Loss: {loss:.4f}, "
                        f"Train: {100 * train_rocauc:.2f}%, "
                        f"Valid: {100 * valid_rocauc:.2f}% "
                        f"Test: {100 * test_rocauc:.2f}%"
                    )
                    with open("log_gnn2.txt", "a") as f:
                        f.write(
                            f"Run: {run + 1:02d}, "
                            f"Epoch: {epoch:02d}, "
                            f"Loss: {loss:.4f}, "
                            f"Train: {100 * train_rocauc:.2f}%, "
                            f"Valid: {100 * valid_rocauc:.2f}% "
                            f"Test: {100 * test_rocauc:.2f}%\n"
                        )

        logger.print_statistics(run)

        # save model
        torch.save(model.state_dict(), "gnn_proteins.pt")

    logger.print_statistics()


def transfer_product_to_protein():
    model = GCN(data_protein.num_features, hidden_channels, 47, num_layers, dropout).to(
        device
    )

    adj_t = data.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t

    pretrained_model = GCN(100, hidden_channels, 112, num_layers, dropout)

    pretrained_model.load_state_dict(torch.load("gnn_product.pt"))

    for param, pretrained_param in zip(
        model.convs[1].parameters(), pretrained_model.convs[1].parameters()
    ):
        param.data = pretrained_param.data.clone()  # Clone the parameter data
        param.requires_grad = False  # Freeze the parameter

    data = data.to(device)

    evaluator = Evaluator(name="ogbn-proteins")
    logger = Logger(3)

    for run in range(3):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, 1 + epochs):
            loss = train(model, data, train_idx_protein, optimizer)

            if epoch % eval_steps == 0:
                result = test(model, data, split_idx_protein, evaluator)
                logger.add_result(run, result)

                if epoch % log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(
                        f"Run: {run + 1:02d}, "
                        f"Epoch: {epoch:02d}, "
                        f"Loss: {loss:.4f}, "
                        f"Train: {100 * train_rocauc:.2f}%, "
                        f"Valid: {100 * valid_rocauc:.2f}% "
                        f"Test: {100 * test_rocauc:.2f}%"
                    )
                    with open("log_gnn2.txt", "a") as f:
                        f.write(
                            f"Run: {run + 1:02d}, "
                            f"Epoch: {epoch:02d}, "
                            f"Loss: {loss:.4f}, "
                            f"Train: {100 * train_rocauc:.2f}%, "
                            f"Valid: {100 * valid_rocauc:.2f}% "
                            f"Test: {100 * test_rocauc:.2f}%\n"
                        )

        logger.print_statistics(run)

        # save model
        torch.save(model.state_dict(), "gnn_proteins.pt")

    logger.print_statistics()


def train(model, data, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval(
        {
            "y_true": data.y[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["rocauc"]
    valid_rocauc = evaluator.eval(
        {
            "y_true": data.y[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["rocauc"]
    test_rocauc = evaluator.eval(
        {
            "y_true": data.y[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["rocauc"]

    return train_rocauc, valid_rocauc, test_rocauc


if __name__ == "__main__":
    train_product()
    train_protein()
    transfer_protein_to_product()
    transfer_product_to_protein()
