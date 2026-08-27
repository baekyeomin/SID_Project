import math
import torch

from torch import Tensor
from typing import Tuple


def gumbel(
    shape: Tuple,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-10,
) -> Tensor:

    uniform = torch.rand(
        shape,
        device=device,
        dtype=dtype,
    )

    uniform = uniform.clamp(
        min=eps,
        max=1.0 - eps,
    )

    return -torch.log(
        -torch.log(uniform)
    )


def gumbel_softmax(
    logits: Tensor,
    temperature: float,
    device: torch.device,
    hard: bool = True,
) -> Tensor:

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than 0."
        )

    noise = gumbel(
        shape=tuple(logits.shape),
        device=device,
        dtype=logits.dtype,
    )

    y_soft = torch.softmax(
        (logits + noise) / temperature,
        dim=-1,
    )

    if not hard:
        return y_soft

    index = y_soft.argmax(
        dim=-1,
        keepdim=True,
    )

    y_hard = torch.zeros_like(
        y_soft
    ).scatter_(
        -1,
        index,
        1.0,
    )

    return (
        y_hard
        - y_soft.detach()
        + y_soft
    )


class TemperatureScheduler:

    def __init__(
        self,
        t0: float = 1.0,
        min_t: float = 0.1,
        anneal_rate: float = 5.8e-5,
    ) -> None:

        if t0 <= 0:
            raise ValueError(
                "t0 must be greater than 0."
            )

        if min_t <= 0:
            raise ValueError(
                "min_t must be greater than 0."
            )

        if min_t > t0:
            raise ValueError(
                "min_t must be less than or equal to t0."
            )

        if anneal_rate < 0:
            raise ValueError(
                "anneal_rate must be greater than or equal to 0."
            )

        self.t0 = t0
        self.min_t = min_t
        self.anneal_rate = anneal_rate

    def get_t(
        self,
        iteration: int,
    ) -> float:

        temperature = (
            self.t0
            * math.exp(
                -self.anneal_rate
                * iteration
            )
        )

        return max(
            temperature,
            self.min_t,
        )