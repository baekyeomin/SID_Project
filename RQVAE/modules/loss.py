from typing import NamedTuple

from torch import nn
from torch import Tensor


# ============================================================
# Reconstruction Loss
# ============================================================

class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat: Tensor, x: Tensor,) -> Tensor:
        return ((x_hat - x) ** 2).sum(dim=-1)


# ============================================================
# Quantize Loss Output
# ============================================================

class QuantizeLossOutput(NamedTuple):
    codebook_loss: Tensor
    commitment_loss: Tensor


# ============================================================
# Quantization Loss
# ============================================================

class QuantizeLoss(nn.Module):
    """
    각 quantization level에서 codebookloss, commitmentloss 따로 계산해서 합함

    여기서는 가중치를 곱하지 않는다!!
    lambda는 최종 RqVaeLoss에서 적용
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, query: Tensor, value: Tensor,) -> QuantizeLossOutput:

        # ----------------------------------------------------
        # Codebook Loss:
        # query gradient 차단 (.detach가 stop gradient)
        # → codebook vector가 query 쪽으로 이동
        # ----------------------------------------------------

        codebook_loss = ((query.detach() - value) ** 2).sum(dim=-1)

        # ----------------------------------------------------
        # Commitment Loss
        # codebook gradient 차단
        # → query / encoder가 code 쪽으로 이동
        # ----------------------------------------------------

        commitment_loss = ((query - value.detach()) ** 2).sum(dim=-1)

        return QuantizeLossOutput(codebook_loss=codebook_loss, commitment_loss=commitment_loss,
        )


# ============================================================
# Final RQ-VAE Loss
# ============================================================

class RqVaeLoss(nn.Module):
    def __init__(self, lambda_rec: float = 1.0, lambda_cb: float = 1.0, lambda_com: float = 0.25,) -> None:

        super().__init__()
				#gin에서 값불러옴
        self.lambda_rec = lambda_rec
        self.lambda_cb = lambda_cb
        self.lambda_com = lambda_com

    def forward(self,reconstruction_loss: Tensor, codebook_loss: Tensor, commitment_loss: Tensor,) -> Tensor:

        return (self.lambda_rec * reconstruction_loss
            + self.lambda_cb * codebook_loss
            + self.lambda_com * commitment_loss
        )