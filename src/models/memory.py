"""
GRU recurrent memory module.

Maintains the agent's internal state across timesteps.
This hidden state is the "cognitive map" that we probe for spatial content.

Member B is responsible for this module.
"""

import torch
import torch.nn as nn


class RecurrentMemory(nn.Module):
    """GRU-based recurrent memory for the navigation agent.

    At each step, takes the CNN feature vector and previous hidden state,
    produces an updated hidden state. This hidden state is the primary
    object of our probing analysis (H1, H2, H3).

    Args:
        input_dim: Dimension of the CNN feature vector.
        hidden_size: GRU hidden state dimension.
        num_layers: Number of stacked GRU layers.
    """

    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) feature vector at current timestep.
            hidden: (num_layers, B, hidden_size) previous hidden state.
                    None initialises to zeros.

        Returns:
            output: (B, hidden_size) GRU output at current timestep.
            hidden: (num_layers, B, hidden_size) updated hidden state.
        """
        # Add time dimension: (B, D) → (B, 1, D)
        x = x.unsqueeze(1)

        if hidden is None:
            hidden = self.init_hidden(x.shape[0], x.device)

        output, hidden = self.gru(x, hidden)
        output = output.squeeze(1)  # (B, 1, H) → (B, H)

        return output, hidden

    def forward_sequence(
        self, x: torch.Tensor, hidden: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a full sequence at once.

        Args:
            x: (B, T, input_dim) feature vectors over T timesteps.
            hidden: (num_layers, B, hidden_size) initial hidden state.

        Returns:
            output: (B, T, hidden_size) GRU outputs at all timesteps.
            hidden: (num_layers, B, hidden_size) final hidden state.
        """
        if hidden is None:
            hidden = self.init_hidden(x.shape[0], x.device)

        output, hidden = self.gru(x, hidden)
        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialise hidden state to zeros."""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
