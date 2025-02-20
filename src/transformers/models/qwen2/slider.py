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

        # Define output size: 2 (for key & value) * token_dim * number of slider heads
        self.output_size = 2 * self.n_token_dim * self.n_slider_heads

        # Define a prefix encoder for each slider variable
        # This maps a scalar slider value to key-value pairs for attention
        self.prefix_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.output_size),  # Maps single scalar input to full-size output
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.output_size, self.n_hidden),  # Hidden layer
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.n_hidden, self.output_size)  # Final projection back to output size
            ) for _ in range(self.n_variables)
        ])

        # Ensure the last linear layer in each prefix encoder produces zero outputs by default
        # This prevents sliders from affecting attention in an untrained state
        for i_var in range(self.n_variables):
            # Tag slider layers to exclude from weight default initialization
            last_layer = self.prefix_encoders[i_var][-1]
            last_layer.last_layer_in_slider = True

    def forward(self, prefix: torch.Tensor):
        """
        Forward pass for generating key-value pairs from slider variables.

        Args:
            prefix (Tensor): Input slider values of shape [batch_size, n_variables].

        Returns:
            Tensor: Key-value pairs of shape [2, batch_size, n_base_heads, seq_len, n_token_dim].
        """

        # Move input to the same device and dtype as the model parameters
        device = next(self.prefix_encoders[0][-1].parameters()).device
        dtype = next(self.prefix_encoders[0][-1].parameters()).dtype
        prefix = prefix.to(device=device, dtype=dtype)

        # Compute key-value representations for each slider variable
        # slider_kv: [batch_size, n_variables, output_size]
        slider_kv = torch.stack(
            [self.prefix_encoders[i_var](prefix[:, i_var:i_var + 1])  # Pass scalar slider values
             for i_var in range(self.n_variables)],
            dim=1
        )

        # Reshape: [batch_size, n_variables, output_size] =>
        #          [batch_size, n_variables, 2 (K & V), n_slider_heads, n_token_dim]
        slider_kv = slider_kv.reshape(
            prefix.shape[0],  # batch_size
            self.n_variables,  # Number of slider variables
            2,  # Two outputs: keys (K) and values (V)
            self.n_slider_heads,  # Slider-specific attention heads
            self.n_token_dim  # Token embedding dimension
        )

        # Expand across attention heads to match `n_base_heads`
        # Repeat each slider head across `n_heads_sharing_slider` heads
        slider_kv = slider_kv.repeat_interleave(self.n_heads_sharing_slider, dim=3)

        # Reshape to match transformer attention format:
        # [2 (K/V), batch_size, n_base_heads, seq_len, n_token_dim]
        slider_kv = slider_kv.permute(2, 0, 3, 1, 4)

        return slider_kv
