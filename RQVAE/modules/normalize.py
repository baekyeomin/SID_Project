#Encoder가 만든 latent vector h(a)의 “길이”를 그대로 둘지, 강제로 1로 맞출지에 따라 encdoer.py에서 False True조정하기

from torch import nn
from torch import Tensor
from torch.nn import functional as F


def l2norm(x, dim=-1, eps=1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=-1, eps=1e-12) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x) -> Tensor:
        return l2norm(x, dim=self.dim, eps=self.eps)
