import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size)
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.Wh = nn.Linear(input_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        z = torch.sigmoid(self.Wz(x) + self.Uz(h))
        r = torch.sigmoid(self.Wr(x) + self.Ur(h))
        h_tilde = torch.tanh(self.Wh(x) + self.Uh(r * h))
        h_next = (1 - z) * h + z * h_tilde
        return h_next

class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        h = torch.zeros(inputs.size(0), self.hidden_size, device=inputs.device)
        for t in range(inputs.size(1)):
            h = self.cell(inputs[:, t, :], h)
        return h