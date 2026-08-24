from typing import NamedTuple

from torch import nn
from torch import Tensor


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat: Tensor, x: Tensor,) -> Tensor:
        return ((x_hat - x) ** 2).sum(dim=-1)


class QuantizeLossOutput(NamedTuple):
    codebook_loss: Tensor
    commitment_loss: Tensor


class QuantizeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, query: Tensor, value: Tensor,) -> QuantizeLossOutput:
        codebook_loss = ((query.detach() - value) ** 2).sum(dim=-1)

        commitment_loss = ((query - value.detach()) ** 2).sum(dim=-1)

        return QuantizeLossOutput(
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )


class RqVaeLoss(nn.Module):
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_cb: float = 1.0,
        lambda_com: float = 0.25,
    ) -> None:
        super().__init__()

        self.lambda_rec = lambda_rec
        self.lambda_cb = lambda_cb
        self.lambda_com = lambda_com

    def forward(
        self,
        reconstruction_loss: Tensor,
        codebook_loss: Tensor,
        commitment_loss: Tensor,
    ) -> Tensor:
        return (
            self.lambda_rec * reconstruction_loss
            + self.lambda_cb * codebook_loss
            + self.lambda_com * commitment_loss
        )