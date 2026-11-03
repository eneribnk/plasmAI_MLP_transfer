import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Multilayer Perceptron (MLP) for plasma surrogate modeling.
    Fully configurable input, hidden, and output sizes.
    Defaults to linear output (no output activation), proven optimal across Ar, O2, and C4F8.
    """
    def __init__(self, input_size=4, output_size=10, h1=20, num_layers=3, freeze_layers=None,
                 output_activation=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.h1 = h1
        self.num_layers = num_layers
        self.output_activation = output_activation
        self.layers = nn.ModuleList()

        # First hidden layer
        self.layers.append(nn.Linear(input_size, h1))

        # Subsequent hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(h1, h1))

        # Output layer
        self.out = nn.Linear(h1, output_size)

        if freeze_layers:
            self.set_frozen_layers(freeze_layers)

    def set_frozen_layers(self, freeze_layers):
        """
        Freeze specific layers by index.
        Indices 0 .. num_layers-1 correspond to hidden layers.
        'out', -1, or num_layers corresponds to the output layer self.out.
        """
        for layer_idx in freeze_layers:
            if isinstance(layer_idx, int) and 0 <= layer_idx < len(self.layers):
                for param in self.layers[layer_idx].parameters():
                    param.requires_grad = False
            elif layer_idx in ('out', -1, len(self.layers)):
                for param in self.out.parameters():
                    param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters in the network."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.relu(self.layers[i](x))
        x = self.out(x)
        if self.output_activation == "elu":
            x = F.elu(x)
        elif self.output_activation == "relu":
            x = F.relu(x)
        # By default (output_activation is None or 'none'): purely linear output layer
        return x


class CombinedModel(nn.Module):
    """
    Combined multi-head architecture for S4 (Geometry Layer strategy).
    Takes a donor hidden layer from one pretrained model and frozen output heads from 10 models.
    """
    def __init__(self, h1=20, num_layers=3, pretrained_models=None, donor_model_idx=6, donor_layer_idx=2):
        super().__init__()
        if pretrained_models is None or len(pretrained_models) == 0:
            raise ValueError("pretrained_models list must be provided for CombinedModel")

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # First layer (4 inputs)
        self.layers.append(nn.Linear(4, h1))

        # Add hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(h1, h1))

        # Donor layer from specified pretrained model (default index 6 -> Point 7)
        if donor_model_idx is not None and donor_layer_idx is not None:
            self.layers[donor_layer_idx] = pretrained_models[donor_model_idx].layers[donor_layer_idx]
            for param in self.layers[donor_layer_idx].parameters():
                param.requires_grad = False

        # Pretrained output heads (frozen)
        self.pretrained_layers = nn.ModuleList([m.out for m in pretrained_models])
        for layer in self.pretrained_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.silu(self.layers[i](x))

        last_layer_outputs = [layer(x) for layer in self.pretrained_layers]
        combined_output = torch.cat(last_layer_outputs, dim=-1)
        return combined_output
