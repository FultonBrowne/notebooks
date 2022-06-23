import torch
from torch import nn
from mixture_of_experts import MoE

# a 3 layered MLP as the experts

class Experts(nn.Module):
    def __init__(self, dim, num_experts = 16):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, dim * 4))
        self.w2 = nn.Parameter(torch.randn(num_experts, dim * 4, dim * 4))
        self.w3 = nn.Parameter(torch.randn(num_experts, dim * 4, dim))
        self.act = nn.LeakyReLU(inplace = True)

    def forward(self, x):
        hidden1 = self.act(torch.einsum('end,edh->enh', x, self.w1))
        hidden2 = self.act(torch.einsum('end,edh->enh', hidden1, self.w2))
        out = torch.einsum('end,edh->enh', hidden2, self.w3)
        return out

experts = Experts(512, num_experts = 16)

moe = MoE(
    dim = 512,
    num_experts = 16,
    experts = experts
)

inputs = torch.randn(4, 1024, 512)
out, aux_loss = moe(inputs) # (4, 1024, 512), (1,)
