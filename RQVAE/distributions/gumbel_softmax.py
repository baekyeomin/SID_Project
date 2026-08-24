import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple


def gumbel(
    shape: Tuple,
    device: torch.device,
    eps: float = 1e-20,
) -> Tensor:
    # Gumbel(0, 1) noise 생성
    uniform = torch.rand(shape, device=device)

    return -torch.log(
        -torch.log(uniform + eps) + eps
    )


def gumbel_softmax(
    logits: Tensor,
    temperature: float,
    device: torch.device,
) -> Tensor:
    # logits에 Gumbel noise를 더해 soft code selection 수행
    noisy_logits = logits + gumbel(
        logits.shape,
        device,
    )

    return F.softmax(
        noisy_logits / temperature,
        dim=-1,
    )


class TemperatureScheduler:
    def __init__(
        self,
        t0: float = 1.0,
        min_t: float = 0.1,
        anneal_rate: float = 5.8e-5,
    ) -> None:
        self.t0 = t0  # 학습 시작 temperature
        self.min_t = min_t  # temperature 최솟값
        self.anneal_rate = anneal_rate  # temperature 감소 속도

    def get_t(
        self,
        iteration: int,
    ) -> float:
        # T(iter) = max(t0 * exp(-rate * iter), min_t)
        temperature = (
            self.t0
            * torch.exp(
                torch.tensor(
                    -self.anneal_rate * iteration,
                    dtype=torch.float32,
                )
            ).item()
        )

        return max(
            temperature,
            self.min_t,
        )