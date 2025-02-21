import torch
import torch.nn as nn


class SliderModel(nn.Module):
    def __init__(self, n_variables, n_hidden, n_heads_sharing_slider, dropout,
                 n_base_heads, n_token_dim):
        """
        A model that encodes slider variables into attention key-value pairs.

        Args:
            n_variables (int): Number of slider variables.
            n_hidden (int): Hidden layer size in the prefix encoder.
            n_heads_sharing_slider (int): Number of base heads sharing one slider head.
            dropout (float): Dropout rate in the prefix encoder.
            n_base_heads (int): Total number of attention heads in the transformer.
            n_token_dim (int): Embedding dimension per token.
        """
        super().__init__()

        # Store model parameters
        self.n_variables = n_variables
        self.n_hidden = n_hidden
        self.n_heads_sharing_slider = n_heads_sharing_slider
        self.dropout = dropout

        # Ensure that the number of base heads is evenly divided by the slider-sharing factor
        self.n_base_heads = n_base_heads
        self.n_token_dim = n_token_dim
        assert self.n_base_heads % self.n_heads_sharing_slider == 0, \
            "n_base_heads must be divisible by n_heads_sharing_slider."

        # Compute the number of slider-specific attention heads
        self.n_slider_heads = self.n_base_heads // self.n_heads_sharing_slider

        # Define output size for key and value separately
        self.kv_size = self.n_token_dim * self.n_slider_heads

        # Define separate prefix encoders for keys and values
        self.key_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.kv_size),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.kv_size, self.n_hidden),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.n_hidden, self.kv_size)  # Final output for keys
            ) for _ in range(self.n_variables)
        ])

        self.value_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.kv_size),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.kv_size, self.n_hidden),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.n_hidden, self.kv_size)  # Final output for values
            ) for _ in range(self.n_variables)
        ])

        # Define attention factor
        self.attention_factor = nn.Parameter(torch.tensor(0.0))

    def forward(self, prefix: torch.Tensor):
        """
        Forward pass for generating key-value pairs from slider variables.

        Args:
            prefix (Tensor): Input slider values of shape [batch_size, n_variables].

        Returns:
            Tensor: Key-value pairs of shape [2, batch_size, n_base_heads, seq_len, n_token_dim].
        """

        # Move input to the same device and dtype as the model parameters
        device = next(self.key_encoders[0][-1].parameters()).device
        dtype = next(self.key_encoders[0][-1].parameters()).dtype
        prefix = prefix.to(device=device, dtype=dtype)

        # Compute key representations for each slider variable
        slider_keys = torch.stack(
            [self.key_encoders[i_var](prefix[:, i_var:i_var + 1])  # Pass scalar slider values
             for i_var in range(self.n_variables)],
            dim=1  # Stack along the variable dimension
        )

        # Compute value representations for each slider variable
        slider_values = torch.stack(
            [self.value_encoders[i_var](prefix[:, i_var:i_var + 1])  # Pass scalar slider values
             for i_var in range(self.n_variables)],
            dim=1
        )

        # Reshape both keys and values to match attention dimensions
        # Shape: [batch_size, n_variables, n_slider_heads, n_token_dim]
        slider_keys = slider_keys.reshape(prefix.shape[0], self.n_variables, self.n_slider_heads, self.n_token_dim)
        slider_values = slider_values.reshape(prefix.shape[0], self.n_variables, self.n_slider_heads, self.n_token_dim)

        # Expand across attention heads to match `n_base_heads`
        # Repeat each slider head across `n_heads_sharing_slider` heads
        slider_keys = slider_keys.repeat_interleave(self.n_heads_sharing_slider, dim=2)
        slider_values = slider_values.repeat_interleave(self.n_heads_sharing_slider, dim=2)

        # Final shape: [batch_size, n_base_heads, seq_len, n_token_dim]
        slider_keys = slider_keys.permute(0, 2, 1, 3)
        slider_values = slider_values.permute(0, 2, 1, 3)
        return slider_keys, slider_values, self.attention_factor
