import torch

class SimpleMetaNeuralNetwork(torch.nn.Module):

    def __init__(self, in_features=27, hidden_size=64, out_classes=3):
        super(SimpleMetaNeuralNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.out_classes = out_classes

        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 2 * hidden_size)
        self.fc3 = torch.nn.Linear(2 * hidden_size, 4 * hidden_size)
        self.fc4 = torch.nn.Linear(4 * hidden_size, out_classes)

        self.relu = torch.nn.ReLU()
        self.all_layers = torch.nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3,
            self.relu,
            self.fc4
        )

    def forward(self, x):
        return self.all_layers(x)

    def train_(self, X, y, optimizer, loss_fn, dataloader, epochs=10):
        self.all_layers.train()
        for epoch in range(epochs):
            for x, y_true in dataloader:
                optimizer.zero_grad()
                y_pred = self.all_layers(x)
                loss = loss_fn(y_pred, y_true)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.all_layers.eval()
        y_pred = self.all_layers(torch.Tensor(X))
        y_pred = torch.nn.functional.sigmoid(y_pred)
        return y_pred.detach().numpy()

    def fit(self, X, y):
        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(y))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(self.all_layers.parameters(), lr=0.001)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        self.train_(X, y, optimizer, loss_fn, train_dataloader, epochs=10)
        return self

    def get_params(self, deep=True):
        return {"in_features": self.fc1.in_features, "hidden_size": self.hidden_size, "out_classes": self.out_classes}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __repr__(self):
        return "SimpleMetaNeuralNetwork(in_features={}, hidden_size={}, out_classes={})".format(self.fc1.in_features, self.hidden_size, self.out_classes)

    def __str__(self):
        return "SimpleMetaNeuralNetwork(in_features={}, hidden_size={}, out_classes={})".format(self.fc1.in_features, self.hidden_size, self.out_classes)
