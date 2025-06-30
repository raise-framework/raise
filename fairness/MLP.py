
import torch
import torch.nn as nn


torch.manual_seed(7)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Initialize weights uniformly in [-1/√fan_in, +1/√fan_in]
        nn.init.uniform_(self.fc1.weight,
                         a=-1.0 / (input_dim**0.5),
                         b=+1.0 / (input_dim**0.5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc2.weight,
                         a=-1.0 / (hidden_dim**0.5),
                         b=+1.0 / (hidden_dim**0.5))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # raw logits

