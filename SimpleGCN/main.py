import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) model.
    
    Args:
        num_features (int): Number of input features.
        num_classes (int): Number of output classes.
    """

    def __init__(self, num_features, num_classes):
        """
        conv1: First graph convolutional layer.
        conv2: Second graph convolutional layer.
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): Input features.
            edge_index (LongTensor): Graph edge indices.
        """

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.conv2(h, edge_index)
        return F.log_softmax(h, dim=1)


def main():
    num_epochs = 1000
    learning_rate = 1e-3

    dataset = Planetoid(root="data/Planetoid", name="Cora")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(dataset.num_features, dataset.num_classes).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train() # Set model to training mode.
    for epoch in range(num_epochs):
        optimizer.zero_grad() # Clear gradients.
        out = model(data.x, data.edge_index) # Perform a single forward pass.
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) # Compute the loss solely based on the training nodes.
        loss.backward() # Derive gradients.
        optimizer.step() # Update parameters based on gradients.

        print(f"Epoch: {epoch + 1:03d}, Loss: {loss.item():.4f}")

    model.eval() # Set model to evaluation mode.
    _, pred = model(data.x, data.edge_index).max(dim=1) # Perform a single forward pass.
    correct = pred[data.test_mask] == data.y[data.test_mask] # Check against ground-truth labels.
    acc = int(correct.sum()) / int(data.test_mask.sum()) # Derive ratio of correct predictions.
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()