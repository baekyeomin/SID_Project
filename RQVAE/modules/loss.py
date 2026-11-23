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


class UniquenessLoss(nn.Module):
    """
    HiD-VAE Uniqueness Loss (DUL)

    batch 안에서 (c1,c2,c3)가 완전히 동일한 쌍(=collision pair)에 대해서만,
    quantization 이전 latent h(a)의 cosine similarity를 margin 아래로
    누르는 margin-based penalty.

    batch 안에 collision이 하나도 없으면 loss = 0
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, z0: Tensor, sem_ids: Tensor) -> Tensor:
        # z0: [B, embed_dim] -> quantization 이전 h(a)
        # sem_ids: [B, 3]    -> (c1, c2, c3)

        z0_normalized = nn.functional.normalize(
            z0,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        cos_sim = z0_normalized @ z0_normalized.T  # [B, B]

        id_match = (
            sem_ids.unsqueeze(1) == sem_ids.unsqueeze(0)
        ).all(dim=-1)  # [B, B], (c1,c2,c3) 전부 같으면 True

        id_match.fill_diagonal_(False)  # 자기 자신 제외

        num_colliding_pairs = id_match.sum()

        if num_colliding_pairs == 0:
            return z0.new_zeros(())

        hinge = (cos_sim - self.margin).clamp(min=0.0)

        return (
            (hinge * id_match.float()).sum()
            / num_colliding_pairs.float()
        )


class RqVaeLoss(nn.Module):
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_cb: float = 1.0,
        lambda_com: float = 0.25,
        lambda_uniq: float = 0.1,
    ) -> None:
        super().__init__()

        self.lambda_rec = lambda_rec
        self.lambda_cb = lambda_cb
        self.lambda_com = lambda_com
        self.lambda_uniq = lambda_uniq

    def forward(
        self,
        reconstruction_loss: Tensor,
        codebook_loss: Tensor,
        commitment_loss: Tensor,
        uniqueness_loss: Tensor,
    ) -> Tensor:
        return (
            self.lambda_rec * reconstruction_loss
            + self.lambda_cb * codebook_loss
            + self.lambda_com * commitment_loss
            + self.lambda_uniq * uniqueness_loss
        )