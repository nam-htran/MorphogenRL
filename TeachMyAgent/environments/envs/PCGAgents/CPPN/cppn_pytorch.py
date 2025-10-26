import torch
import torch.nn as nn
import numpy as np
import os


class CPPN_Pytorch(nn.Module):
    """
    PyTorch version of TanHSoftplusMixCPPN.
    - Matches the original TensorFlow model (no bias terms).
    - Supports loading pretrained weights converted from TensorFlow.
    """
    def __init__(self, x_dim: int, input_dim: int, output_dim: int = 2):
        super(CPPN_Pytorch, self).__init__()
        self.x_dim = x_dim
        self.input_dim = input_dim

        # Important: all layers use `bias=False` to match the original model
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 64, bias=False),
            nn.Tanh(),
            nn.Linear(64, 64, bias=False),
            nn.Softplus(),
            nn.Linear(64, 64, bias=False),
            nn.Tanh(),
            nn.Linear(64, 64, bias=False),
            nn.Softplus(),
            nn.Linear(64, output_dim, bias=False)
        )
        print("CPPN (PyTorch) initialized — no bias parameters.")

    def load_tf_weights(self, weights_path: str) -> None:
        """Load pretrained weights (converted from TensorFlow)."""
        if not os.path.exists(weights_path):
            print(f"Weight file '{weights_path}' not found. Using random initialization.")
            print("Run 'convert_weights.py' to generate the .pt file.")
            return

        state_dict = torch.load(weights_path)
        self.net.load_state_dict(state_dict)
        print("Successfully loaded original TensorFlow weights into CPPN (PyTorch).")

    def generate(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Generate terrain data from an input vector.

        Args:
            input_vector: NumPy array representing the latent input.

        Returns:
            NumPy array of generated terrain values.
        """
        input_vector_t = torch.from_numpy(input_vector).float()

        x = np.arange(self.x_dim)
        scaled_x = x / (self.x_dim - 1)
        x_vec_t = torch.from_numpy(scaled_x).float().reshape((self.x_dim, 1))

        # Repeat input vector across the x-axis and concatenate
        reshaped_input = input_vector_t.repeat(self.x_dim, 1)
        final_input = torch.cat((x_vec_t, reshaped_input), dim=1)

        with torch.no_grad():
            output = self.net(final_input)

        return output.numpy()
