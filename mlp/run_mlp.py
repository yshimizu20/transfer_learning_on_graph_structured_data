import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from mlp import *
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


def train_protein():
    model = MLP(x_protein.size(-1), hidden_channels, 112, num_layers, dropout).to(
        device
    )

    evaluator = Evaluator(name="ogbn-proteins")
    logger = Logger(3)

    for run in range(3):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, 1 + epochs):
            loss = train(model, x_protein, y_protein_true, train_idx_protein, optimizer)

            if epoch % eval_steps == 0:
                result = test(
                    model, x_protein, y_protein_true, split_idx_protein, evaluator
                )
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
                    with open("log_mlp5.txt", "a") as f:
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
        torch.save(model.state_dict(), "mlp_proteins.pt")

    logger.print_statistics()


def train_product():
    model = MLP(x_product.size(-1), hidden_channels, 47, num_layers, dropout).to(device)

    evaluator = Evaluator(name="ogbn-products")
    logger = Logger(3)

    for run in range(3):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, 1 + epochs):
            loss = train(model, x_product, y_product_true, train_idx_product, optimizer)

            if epoch % eval_steps == 0:
                result = test(
                    model, x_product, y_product_true, split_idx_product, evaluator
                )
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
                    with open("log_mlp5.txt", "a") as f:
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
        torch.save(model.state_dict(), "mlp_products.pt")

    logger.print_statistics()


def transfer_product_to_protein():
    model = MLP(x_protein.size(-1), 256, 112, 3, 0.5).to(device)

    pretrained_model = MLP(100, 256, 47, 3, 0.5).to(device)
    pretrained_model.load_state_dict(torch.load("mlp_products.pt"))

    logger = Logger(3)
    evaluator = Evaluator(name="ogbn-proteins")

    for run in range(3):
        model.reset_parameters()

        for param, pretrained_param in zip(
            model.lins[1].parameters(), pretrained_model.lins[1].parameters()
        ):
            param.data = pretrained_param.data.clone()  # Copy weights
            param.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, 1 + epochs):
            loss = train(model, x_protein, y_protein_true, train_idx_protein, optimizer)

            if epoch % eval_steps == 0:
                result = test(
                    model, x_protein, y_protein_true, split_idx_protein, evaluator
                )
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
                    with open("log_mlp_4layers.txt", "a") as f:
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
        torch.save(model.state_dict(), "mlp_proteins_transfer.pt")

    logger.print_statistics()


def transfer_protein_to_product():
    model = MLP(100, 256, 47, 3, 0.5).to(device)

    pretrained_model = MLP(x_product.size(-1), 256, 112, 3, 0.5).to(device)
    pretrained_model.load_state_dict(torch.load("mlp_proteins.pt"))

    logger = Logger(3)
    evaluator = Evaluator(name="ogbn-products")

    for run in range(3):
        model.reset_parameters()

        for param, pretrained_param in zip(
            model.lins[1].parameters(), pretrained_model.lins[1].parameters()
        ):
            param.data = pretrained_param.data.clone()  # Copy weights
            param.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, 1 + epochs):
            loss = train(model, x_product, y_product_true, train_idx_product, optimizer)

            if epoch % eval_steps == 0:
                result = test(
                    model, x_product, y_product_true, split_idx_product, evaluator
                )
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
                    with open("log_mlp_4layers.txt", "a") as f:
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
        torch.save(model.state_dict(), "mlp_products_transfer.pt")

    logger.print_statistics()


def train(model, x, y_true, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(x)[train_idx]
    loss = criterion(out, y_true[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y_true, split_idx, evaluator):
    model.eval()

    y_pred = model(x)

    train_rocauc = evaluator.eval(
        {
            "y_true": y_true[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["rocauc"]
    valid_rocauc = evaluator.eval(
        {
            "y_true": y_true[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["rocauc"]
    test_rocauc = evaluator.eval(
        {
            "y_true": y_true[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["rocauc"]

    return train_rocauc, valid_rocauc, test_rocauc


if __name__ == "__main__":
    train_protein()
    train_product()
    transfer_product_to_protein()
    transfer_protein_to_product()
