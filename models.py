import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_dim=2, output_dim = 1, hidden_layers=[256, 128, 64]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_layers[0], self.hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_layers[1], self.hidden_layers[2]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_layers[2], self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output
    
    
class Generator(nn.Module):
    def __init__(self, input_dim=2, output_dim = 1, hidden_layers=[16, 32], output_activation=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.output_activation = output_activation
        
         # Create the layers of the Discriminator
        layers = []
        prev_size = input_dim
        for layer_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, layer_size),
                nn.ReLU()
            ])
            prev_size = layer_size

        # Add the output activation function if specified
        if self.output_activation is not None:
            layers.append(nn.Tanh())

        layers.append(nn.Linear(prev_size, self.output_dim))
        self.model = nn.Sequential(*layers)
        '''
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[0], self.hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[2], self.output_dim),
            nn.Tanh(),
        )'''

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output