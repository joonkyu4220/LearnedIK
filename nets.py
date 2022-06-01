import torch
import torch.nn as nn

class IKNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.repr = args.repr

        self.input_dims = [400, 300, 200, 100, 50]
        
        if args.activation == "ReLU":
            self.activation = nn.ReLU()
        elif args.activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif args.activation == "Tanh":
            self.activation = nn.Tanh()
        
        self.dropout = 0.1

        layers = []

        input_dim = 4 if self.repr == "COSSIN" else 3
        
        layers.append(nn.BatchNorm1d(input_dim))

        for output_dim in self.input_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(self.activation)
            
            layers.append(nn.Dropout(self.dropout))
            input_dim = output_dim

        output_dim = 6 if self.repr == "COSSIN" else 3
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class IKNet1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.repr = args.repr

        self.input_dims = [50, 50]

        input_dim = 4 if self.repr == "COSSIN" else 3

        layers = []

        layers.append(nn.BatchNorm1d(input_dim))

        for output_dim in self.input_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.LeakyReLU())
            input_dim = output_dim

        output_dim = 6 if self.repr == "COSSIN" else 3
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FKNet(nn.Module):
    def __init__(self, args):
        super(FKNet, self).__init__()
        self.repr = args.repr
        self.normalize = args.normalize
    
    def normalized(self, rot):
        rot_norm = torch.zeros_like(rot)
        rot_norm[:, 0:2] = rot[:, 0:2] / torch.sqrt(torch.sum(torch.square(rot[:, 0:2]), dim=1, keepdim=True))
        rot_norm[:, 2:4] = rot[:, 2:4] / torch.sqrt(torch.sum(torch.square(rot[:, 2:4]), dim=1, keepdim=True))
        rot_norm[:, 4:6] = rot[:, 4:6] / torch.sqrt(torch.sum(torch.square(rot[:, 4:6]), dim=1, keepdim=True))
        return rot_norm

    def forward(self, rot, lengths):
        if self.repr == "ANGLE":
            q00 = torch.sum(rot[:, 0:1], dim=1, keepdim=True)
            q01 = torch.sum(rot[:, 0:2], dim=1, keepdim=True)
            q02 = torch.sum(rot[:, 0:3], dim=1, keepdim=True)
            c00 = torch.cos(q00)
            s00 = torch.sin(q00)
            c01 = torch.cos(q01)
            s01 = torch.sin(q01)
            c02 = torch.cos(q02)
            s02 = torch.sin(q02)
            cos = torch.cat([c00, c01, c02], dim=1)
            sin = torch.cat([s00, s01, s02], dim=1)
            x = torch.sum(lengths * cos, dim=1, keepdim=True)
            y = torch.sum(lengths * sin, dim=1, keepdim=True)
            return torch.cat([x, y, q02], dim=1)
        elif self.repr == "COSSIN":
            rot_norm = self.normalized(rot) if self.normalize else rot
            c00 = rot_norm[:, 0:1]
            s00 = rot_norm[:, 1:2]
            c01 = c00 * rot_norm[:, 2:3] - s00 * rot_norm[:, 3:4]
            s01 = s00 * rot_norm[:, 2:3] + c00 * rot_norm[:, 3:4]
            c02 = c01 * rot_norm[:, 4:5] - s01 * rot_norm[:, 5:6]
            s02 = s01 * rot_norm[:, 4:5] + c01 * rot_norm[:, 5:6]
            cos = torch.cat([c00, c01, c02], dim=1)
            sin = torch.cat([s00, s01, s02], dim=1)
            x = torch.sum(lengths * cos, dim=1, keepdim=True)
            y = torch.sum(lengths * sin, dim=1, keepdim=True)
            return torch.cat([x, y, c02, s02], dim = 1)


class GatingNet(nn.Module):
    def __init__(self, args):
        super(GatingNet, self).__init__()

        self.repr = args.repr
        self.num_nets = args.num_nets

        self.input_dims = [400, 300, 200, 100, 50]
        self.dropout = 0.1
        layers = []
        input_dim = 7 if self.repr == "COSSIN" else 6
        
        layers.append(nn.BatchNorm1d(input_dim))

        for output_dim in self.input_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(self.dropout))
            input_dim = output_dim
        output_dim = self.num_nets
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)