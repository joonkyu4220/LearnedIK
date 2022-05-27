import torch
import torch.nn as nn

class IKNet(nn.Module):
    def __init__(self, repr):
        super().__init__()
        self.repr = repr

        self.input_dims = [400, 300, 200, 100, 50]
        self.dropout = 0.1

        layers = []

        input_dim = 4 if self.repr == "COSSIN" else 3
        layers.append(nn.BatchNorm1d(input_dim))

        for output_dim in self.input_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            input_dim = output_dim

        output_dim = 6 if self.repr == "COSSIN" else 3
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class FKNet(nn.Module):
    def __init__(self, lengths, repr, normalize):
        super(FKNet, self).__init__()
        self.l0, self.l1, self.l2 = lengths
        self.repr = repr
        self.normalize = normalize
    
    def normalized(self, x):
        x_norm = torch.zeros_like(x)
        x_norm[:, 0:2] = x[:, 0:2] / torch.sqrt(torch.sum(torch.square(x[:, 0:2]), dim=1, keepdim=True))
        x_norm[:, 2:4] = x[:, 2:4] / torch.sqrt(torch.sum(torch.square(x[:, 2:4]), dim=1, keepdim=True))
        x_norm[:, 4:6] = x[:, 4:6] / torch.sqrt(torch.sum(torch.square(x[:, 4:6]), dim=1, keepdim=True))
        return x_norm

    def forward(self, x):
        if self.repr == "ANGLE":
            q0 = torch.sum(x[:, 0:1], dim=1, keepdim=True)
            q01 = torch.sum(x[:, 0:2], dim=1, keepdim=True)
            q02 = torch.sum(x[:, 0:3], dim=1, keepdim=True)
            c00 = torch.cos(q0)
            s00 = torch.sin(q0)
            c01 = torch.cos(q01)
            s01 = torch.sin(q01)
            c02 = torch.cos(q02)
            s02 = torch.sin(q02)
            x = self.l0 * c00 + self.l1 * c01 + self.l2 * c02
            y = self.l0 * s00 + self.l1 * s01 + self.l2 * s02
            return torch.cat([x, y, q02], dim=1)
        elif self.repr == "COSSIN":
            if self.normalize:
                x = self.normalized(x)
            c00 = x[:, 0:1]
            s00 = x[:, 1:2]
            c01 = c00 * x[:, 2:3] - s00 * x[:, 3:4]
            s01 = s00 * x[:, 2:3] + c00 * x[:, 3:4]
            c02 = c01 * x[:, 4:5] - s01 * x[:, 5:6]
            s02 = s01 * x[:, 4:5] + c01 * x[:, 5:6]
            x = self.l0 * c00 + self.l1 * c01 + self.l2 * c02
            y = self.l0 * s00 + self.l1 * s01 + self.l2 * s02
            return torch.cat([x, y, c02, s02], dim = 1)
