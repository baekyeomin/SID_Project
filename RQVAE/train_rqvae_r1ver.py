
"""R1 experiment: initialize Q2 KMeans with event-wise mean r1(a).

All other training logic is inherited from train_rqvae_hver.py.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_rqvae_hver as hver


@torch.no_grad()
def encode_all_train_articles_r1(
    model,
    train_dataset,
    device,
    batch_size=512,
    num_workers=0,
):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    was_training = model.training
    model.eval()
    residuals = []
    event_ids_list = []

    try:
        for batch in tqdm(
            loader,
            desc="Encoding train r1 for Q2 initialization",
        ):
            x, category_ids, event_ids = hver.unpack_batch(batch)

            x = x.to(
                device=device,
                dtype=next(model.encoder.parameters()).dtype,
            )
            category_ids = category_ids.to(
                device=device,
                dtype=torch.long,
            )

            # r1(a) = h(a) - Q1[category(a)]
            r1 = model.get_c2_residual(
                x=x,
                category_ids=category_ids,
            )

            residuals.append(r1.detach().float().cpu())
            event_ids_list.append(event_ids.detach().cpu())
    finally:
        model.train(was_training)

    if not residuals:
        raise ValueError(
            "Train dataset is empty; cannot initialize Q2"
        )

    return (
        torch.cat(residuals, dim=0),
        torch.cat(event_ids_list, dim=0),
    )


def train_r1():
    hver.encode_all_train_articles = encode_all_train_articles_r1

    print("[R1] Q2 KMeans input: event-wise mean r1(a)")
    print("[R1] Checkpoint path: configured by Gin")

    return hver.train(
        pretrained_rqvae_path=None,
    )


if __name__ == "__main__":
    hver.parse_config()
    train_r1()
