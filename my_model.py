import torch
from vol import V_and_MuY_rho_tanh

class NeuralSDEModel(torch.nn.Module):
    def __init__(self):
        super(NeuralSDEModel, self).__init__()
        self.dense1 = torch.nn.Linear(3, 64)
        self.norm1 = torch.nn.LayerNorm(64)
        self.dense2 = torch.nn.Linear(64, 32)
        self.norm2 = torch.nn.LayerNorm(32)
        self.dense3 = torch.nn.Linear(32, 4)
        # self.norm3 = torch.nn.LayerNorm(32)
        # self.dense4 = torch.nn.Linear(32, 4)

    def forward(self, x):
        x = self.dense1(x)
        x = self.norm1(x)
        x = torch.nn.functional.silu(x)  # Swish activation
        x = self.dense2(x)
        x = self.norm2(x)
        x = torch.nn.functional.silu(x)  # Swish activation
        x = self.dense3(x)
        # x = self.norm3(x)
        # x = torch.nn.functional.silu(x)  # Swish activation
        # x = self.dense4(x)
        return x


MODEL = {
    "nets": {
        "phi": NeuralSDEModel().to("cuda:0"),
    },
    "name": "spx_vix_small",
    "optimizer": None,
}

V_AND_MUY = V_and_MuY_rho_tanh
