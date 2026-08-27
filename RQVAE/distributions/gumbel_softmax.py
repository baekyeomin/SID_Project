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

    gumbel_noise = -torch.log(
        -torch.log(uniform)
    )

    return gumbel_noise


def gumbel_softmax(
    logits: Tensor,
    temperature: float,
    device: torch.device,
    hard: bool = True,
) -> Tensor:

    if temperature <= 0:
        raise ValueError(
            f"temperature must be > 0, got {temperature}."
        )

    noise = gumbel(
        shape=tuple(logits.shape),
        device=device,
        dtype=logits.dtype,
    )

    noisy_logits = (
        logits
        + noise
    )

    y_soft = torch.softmax(
        noisy_logits / temperature,
        dim=-1,
    )

    if not hard:
        return y_soft

    max_indices = y_soft.argmax(
        dim=-1,
        keepdim=True,
    )

    y_hard = torch.zeros_like(
        y_soft
    ).scatter_(
        dim=-1,
        index=max_indices,
        value=1.0,
    )

    y = (
        y_hard
        - y_soft.detach()
        + y_soft
    )

    return y


class TemperatureScheduler:
    def __init__(
        self,
        t0: float = 1.0,
        min_t: float = 0.1,
        anneal_rate: float = 5.8e-5,
    ) -> None:

        if t0 <= 0:
            raise ValueError(
                f"t0 must be > 0, got {t0}."
            )

        if min_t <= 0:
            raise ValueError(
                f"min_t must be > 0, got {min_t}."
            )

        if anneal_rate < 0:
            raise ValueError(
                "anneal_rate must be >= 0, "
                f"got {anneal_rate}."
            )

        if min_t > t0:
            raise ValueError(
                "min_t must be <= t0. "
                f"t0={t0}, min_t={min_t}."
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