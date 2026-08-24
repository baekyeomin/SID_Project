import gin  # Gin config 사용
import importlib  # torch 내부 모듈을 동적으로 다시 불러올 때 사용
import os  # 폴더/파일 경로 및 checkpoint 관리
import random  # Python random seed 설정

import numpy as np  # NumPy 연산 및 seed 설정
import torch  # PyTorch
import wandb  # 학습 loss 시각화 및 logging

from accelerate import Accelerator  # mixed precision 및 multi-device 학습 관리
from sklearn.cluster import KMeans  # Q2 초기화를 위한 K-means
from torch.optim import AdamW  # RQ-VAE optimizer
from torch.utils.data import DataLoader  # dataset batch 처리
from tqdm import tqdm  # 진행률 표시

from data.news import NewsArticleDataset  # EB-NeRD 기사 dataset
from modules.rqvae import RqVae  # 전체 RQ-VAE 모델
from modules.quantize import QuantizeForwardMode  # STE/Gumbel/Rotation 방식 지정
from modules.utils import parse_config  # Gin config 파일 읽기


def ensure_torch_serialization_compatibility() -> None:
    # 일부 Colab 환경에서 torch.Tensor 속성이 사라지는 문제 복구
    if not hasattr(torch, "Tensor"):
        tensor_class = None

        try:
            tensor_module = importlib.import_module("torch._tensor")  # torch Tensor 모듈 직접 import
            tensor_class = getattr(tensor_module, "Tensor", None)  # Tensor class 가져오기
        except Exception:
            tensor_class = None

        if tensor_class is None:
            try:
                tensor_class = type(torch.empty(0))  # 실제 Tensor를 생성해 class 추출
            except Exception as exc:
                raise RuntimeError("torch.Tensor attribute를 복구하지 못했습니다.") from exc

        setattr(torch, "Tensor", tensor_class)  # torch.Tensor 속성을 다시 등록

    # 일부 환경에서 torch._utils 속성이 사라지는 문제 복구
    if not hasattr(torch, "_utils"):
        try:
            torch_utils_module = importlib.import_module("torch._utils")  # torch._utils 직접 import
            setattr(torch, "_utils", torch_utils_module)  # torch namespace에 다시 등록
        except Exception as exc:
            raise RuntimeError("torch._utils attribute를 복구하지 못했습니다.") from exc

    # 복구 후 최종 확인
    if not hasattr(torch, "Tensor"):
        raise RuntimeError("torch.Tensor is still unavailable.")
    if not hasattr(torch, "_utils"):
        raise RuntimeError("torch._utils is still unavailable.")


ensure_torch_serialization_compatibility()  # 프로그램 시작 시 serialization 문제 미리 복구


def safe_torch_save(state, path: str) -> None:
    ensure_torch_serialization_compatibility()  # 저장 직전 torch 상태 확인
    temp_path = path + ".tmp"  # 저장 중 오류를 대비한 임시 파일 경로

    if os.path.exists(temp_path):
        os.remove(temp_path)  # 이전 임시 파일이 남아 있으면 삭제

    try:
        torch.save(state, temp_path)  # 먼저 임시 파일에 checkpoint 저장
        os.replace(temp_path, path)  # 저장 성공 후 실제 checkpoint 파일로 교체
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # 실패 시 깨진 임시 파일 제거
        raise  # 원래 오류 다시 발생


def safe_torch_load(path: str, map_location=None):
    ensure_torch_serialization_compatibility()  # checkpoint load 전 torch 상태 확인
    return torch.load(
        path,
        map_location=map_location,  # checkpoint를 불러올 device
        weights_only=False,  # optimizer/config 등 weight 외 정보도 함께 불러오기
    )


def test_torch_serialization(save_dir_root: str) -> None:
    os.makedirs(save_dir_root, exist_ok=True)  # checkpoint 저장 폴더 생성
    test_path = os.path.join(save_dir_root, "_serialization_test.pt")  # 테스트 파일 경로
    test_state = {"tensor": torch.zeros(2, dtype=torch.float32)}  # 간단한 테스트 Tensor 생성

    safe_torch_save(test_state, test_path)  # 실제 저장 테스트
    loaded = safe_torch_load(test_path, map_location="cpu")  # 다시 load 테스트

    if "tensor" not in loaded:
        raise RuntimeError("torch serialization smoke test failed.")  # 정상 load 여부 확인

    if os.path.exists(test_path):
        os.remove(test_path)  # 테스트 파일 삭제

    print("PyTorch checkpoint save/load test: OK")


def set_seed(seed: int) -> None:
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 모든 GPU seed 설정


def unpack_batch(batch):
    x = batch["x"]  # 기사 embedding

    if "category_id" in batch:
        category_ids = batch["category_id"]  # 일반 category ID 사용
    elif "model_category_id" in batch:
        category_ids = batch["model_category_id"]  # 전처리된 model category ID 사용
    else:
        raise KeyError("Batch must contain 'category_id' or 'model_category_id'.")

    if "event_id" not in batch:
        raise KeyError("Batch must contain 'event_id'.")  # event 정보가 없으면 Q2 초기화 불가능

    event_ids = batch["event_id"]  # 기사별 event ID

    return x, category_ids, event_ids  # 학습에 필요한 세 값 반환


def print_train_config(
    iterations,
    batch_size,
    learning_rate,
    weight_decay,
    gradient_accumulate_every,
    dataset_folder,
    vae_input_dim,
    vae_hidden_dims,
    vae_embed_dim,
    vae_num_categories,
    vae_c2_codebook_size,
    vae_c3_codebook_size,
    vae_codebook_normalize,
    vae_sim_vq,
    vae_codebook_mode,
    lambda_rec,
    lambda_cb,
    lambda_com,
    amp,
    mixed_precision_type,
    save_dir_root,
    do_eval,
    seed,
):
    print("\n" + "=" * 70)
    print("RQ-VAE TRAIN CONFIG")
    print("=" * 70)
    print(f"dataset_folder             : {dataset_folder}")

    print("\n[Training]")
    print(f"iterations                 : {iterations}")
    print(f"batch_size                 : {batch_size}")
    print(f"learning_rate              : {learning_rate}")
    print(f"weight_decay               : {weight_decay}")
    print(f"gradient_accumulate_every  : {gradient_accumulate_every}")

    print("\n[Network]")
    print(f"vae_input_dim              : {vae_input_dim}")
    print(f"vae_hidden_dims            : {vae_hidden_dims}")
    print(f"vae_embed_dim              : {vae_embed_dim}")

    print("\n[Codebooks]")
    print(f"Q1 num_categories          : {vae_num_categories}")
    print(f"Q2 codebook size           : {vae_c2_codebook_size}")
    print(f"Q3 codebook size           : {vae_c3_codebook_size}")

    print("\n[Quantization]")
    print(f"codebook_normalize         : {vae_codebook_normalize}")
    print(f"sim_vq                     : {vae_sim_vq}")
    print(f"forward_mode               : {vae_codebook_mode}")

    print("\n[Loss]")
    print(f"lambda_rec                 : {lambda_rec}")
    print(f"lambda_cb                  : {lambda_cb}")
    print(f"lambda_com                 : {lambda_com}")

    print("\n[Runtime]")
    print(f"amp                        : {amp}")
    print(f"mixed_precision_type       : {mixed_precision_type}")
    print(f"do_eval                    : {do_eval}")
    print(f"seed                       : {seed}")
    print(f"save_dir_root              : {save_dir_root}")
    print("=" * 70 + "\n")


@torch.no_grad()
def encode_all_train_articles(
    model: RqVae,
    train_dataset: NewsArticleDataset,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # h(a) 추출 시 Encoder에 한 번에 넣을 기사 수
        shuffle=False,  # 전체 기사 순서를 유지
        drop_last=False,  # 마지막 작은 batch도 포함
        num_workers=num_workers,
    )

    was_training = model.training  # 기존 train/eval 상태 기억
    model.eval()  # h(a) 추출 동안 evaluation mode 사용

    h_list = []  # 모든 기사 h(a)를 모을 리스트
    event_id_list = []  # 각 h(a)에 대응하는 event ID 저장

    for batch in tqdm(loader, desc="Encoding train articles for Q2 initialization"):
        x, _, event_ids = unpack_batch(batch)  # category ID는 Q2 초기화에서 사용하지 않음

        x = x.to(
            device=device,
            dtype=next(model.encoder.parameters()).dtype,  # Encoder와 dtype 일치
        )

        h = model.encode(x)  # 전체 Train 기사에 대해 latent vector h(a) 계산

        h_list.append(h.detach().cpu())  # K-means용으로 CPU에 저장
        event_id_list.append(event_ids.detach().cpu())  # event ID도 CPU에 저장

    h_all = torch.cat(h_list, dim=0)  # 모든 batch의 h(a)를 하나로 합침
    event_ids_all = torch.cat(event_id_list, dim=0)  # 모든 event ID를 하나로 합침

    if was_training:
        model.train()  # 원래 training 상태였다면 다시 복구

    return h_all, event_ids_all  # 전체 Train h(a)와 event ID 반환


@torch.no_grad()
def compute_event_representations(
    h_all: torch.Tensor,
    event_ids_all: torch.Tensor,
):
    unique_event_ids, inverse_indices = torch.unique(
        event_ids_all,
        sorted=True,  # event ID를 정렬
        return_inverse=True,  # 각 기사 event가 unique_event_ids의 몇 번째인지 반환
    )

    num_events = unique_event_ids.shape[0]  # 전체 Train event 개수
    embed_dim = h_all.shape[1]  # latent vector 차원

    event_sums = torch.zeros(
        num_events,
        embed_dim,
        dtype=h_all.dtype,
    )  # event별 h(a) 합을 저장할 Tensor

    event_sums.index_add_(
        0,
        inverse_indices,
        h_all,
    )  # 같은 event의 h(a)를 모두 더함

    event_counts = torch.zeros(
        num_events,
        dtype=h_all.dtype,
    )  # event별 기사 개수를 저장할 Tensor

    ones = torch.ones(
        event_ids_all.shape[0],
        dtype=h_all.dtype,
    )  # 각 article을 1개로 세기 위한 Tensor

    event_counts.index_add_(
        0,
        inverse_indices,
        ones,
    )  # 동일 event의 기사 수 누적

    z_events = event_sums / event_counts.unsqueeze(1).clamp_min(1.0)  # z(E)=event별 전체 h(a) 평균

    return unique_event_ids, z_events  # event ID와 해당 대표 vector 반환


@torch.no_grad()
def initialize_c2_codebook(
    model: RqVae,
    train_dataset: NewsArticleDataset,
    device: torch.device,
    c2_codebook_size: int,
    encode_batch_size: int = 512,
    kmeans_n_init: int = 10,
    seed: int = 42,
):
    print("\n" + "=" * 70)
    print("Q2 CODEBOOK INITIALIZATION")
    print("=" * 70)

    h_all, event_ids_all = encode_all_train_articles(
        model=model,
        train_dataset=train_dataset,
        device=device,
        batch_size=encode_batch_size,  # 전체 Train h(a)를 추출할 때만 사용하는 batch size
    )

    print(f"Train articles : {h_all.shape[0]}")

    unique_event_ids, z_events = compute_event_representations(
        h_all=h_all,
        event_ids_all=event_ids_all,
    )  # 전체 Train 기준으로 event 대표 vector z(E) 계산

    print(f"Train events   : {z_events.shape[0]}")
    print(f"z(E) shape     : {tuple(z_events.shape)}")

    if z_events.shape[0] < c2_codebook_size:
        raise ValueError(
            "Number of train events must be >= C2 codebook size. "
            f"events={z_events.shape[0]}, c2_codebook_size={c2_codebook_size}"
        )  # event 수보다 cluster 수가 많으면 K-means 불가능

    print(f"Running K-means (k={c2_codebook_size})...")

    kmeans = KMeans(
        n_clusters=c2_codebook_size,  # Q2 code 개수와 동일하게 cluster 수 설정
        init="k-means++",  # 초기 centroid를 k-means++ 방식으로 선택
        n_init=kmeans_n_init,  # 서로 다른 초기값으로 K-means를 반복 실행
        random_state=seed,  # 재현성을 위한 seed
    )

    kmeans.fit(z_events.numpy())  # 모든 Train event 대표 vector로 K-means 학습

    centroids = torch.from_numpy(kmeans.cluster_centers_).float()  # centroid를 PyTorch Tensor로 변환

    print(f"Centroid shape : {tuple(centroids.shape)}")

    model.set_c2_codebook(centroids.to(device))  # K-means centroid로 Q2 초기화

    print("Q2 codebook initialized.")
    print("=" * 70 + "\n")

    return unique_event_ids, z_events, centroids


@torch.no_grad()
def evaluate(model, dataloader, device, gumbel_t):
    model.eval()  # evaluation mode로 변경

    total_losses = []  # 전체 loss 저장
    reconstruction_losses = []  # reconstruction loss 저장
    codebook_losses = []  # codebook loss 저장
    commitment_losses = []  # commitment loss 저장
    rqvae_losses = []  # weighted quantization loss 저장

    for batch in dataloader:
        x, category_ids, _ = unpack_batch(batch)  # validation에서는 event ID 사용하지 않음

        x = x.to(device=device, non_blocking=True)  # 기사 embedding GPU 이동
        category_ids = category_ids.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )  # category ID GPU 이동

        output = model(
            x=x,
            category_ids=category_ids,
            gumbel_t=gumbel_t,
        )  # validation forward

        total_losses.append(output.loss.detach().float().cpu().item())  # batch total loss 저장
        reconstruction_losses.append(output.reconstruction_loss.detach().float().cpu().item())  # reconstruction 저장
        codebook_losses.append(output.codebook_loss.detach().float().cpu().item())  # codebook 저장
        commitment_losses.append(output.commitment_loss.detach().float().cpu().item())  # commitment 저장
        rqvae_losses.append(output.rqvae_loss.detach().float().cpu().item())  # quantization loss 저장

    return {
        "loss": np.mean(total_losses),  # validation 전체 평균 total loss
        "reconstruction_loss": np.mean(reconstruction_losses),  # 평균 reconstruction loss
        "codebook_loss": np.mean(codebook_losses),  # 평균 codebook loss
        "commitment_loss": np.mean(commitment_losses),  # 평균 commitment loss
        "rqvae_loss": np.mean(rqvae_losses),  # 평균 weighted quantization loss
    }


def build_checkpoint_state(
    accelerator,
    model,
    optimizer,
    iteration,
    lambda_rec,
    lambda_cb,
    lambda_com,
):
    unwrapped_model = accelerator.unwrap_model(model)  # Accelerate wrapper를 제거한 실제 RQ-VAE 모델

    state = {
        "iter": iteration,  # 저장 시점 iteration
        "model": unwrapped_model.state_dict(),  # Encoder/Decoder/Q1/Q2/Q3 parameter
        "model_config": unwrapped_model.config,  # 모델 구조 복원을 위한 설정
        "optimizer": optimizer.state_dict(),  # optimizer 상태
        "loss_weights": {
            "lambda_rec": lambda_rec,
            "lambda_cb": lambda_cb,
            "lambda_com": lambda_com,
        },  # loss 가중치 저장
        "gin_config": gin.operative_config_str(),  # 실제 적용된 Gin config 저장
    }

    return state


@gin.configurable
def train(
    iterations: int = 50000,  # 전체 optimizer update 횟수
    batch_size: int = 64,  # 실제 학습 batch size
    learning_rate: float = 1e-4,  # AdamW learning rate
    weight_decay: float = 0.01,  # AdamW weight decay
    gradient_accumulate_every: int = 1,  # 몇 batch의 gradient를 누적할지
    dataset_folder: str = "datasets/ebnerd",  # EB-NeRD 데이터 위치
    vae_input_dim: int = 768,  # 원본 기사 embedding 차원
    vae_hidden_dims=[512, 256],  # Encoder hidden dimensions
    vae_embed_dim: int = 128,  # latent/code vector 차원
    vae_num_categories: int = 25,  # Q1 category 수
    vae_c2_codebook_size: int = 256,  # Q2 code 개수
    vae_c3_codebook_size: int = 256,  # Q3 code 개수
    vae_codebook_normalize: bool = False,  # latent/codebook L2 normalization 여부
    vae_sim_vq: bool = False,  # SIM-VQ projection 사용 여부
    vae_codebook_mode=QuantizeForwardMode.STE,  # Q2/Q3 gradient 전달 방식
    lambda_rec: float = 1.0,  # reconstruction loss 가중치
    lambda_cb: float = 1.0,  # codebook loss 가중치
    lambda_com: float = 0.25,  # commitment loss 가중치
    kmeans_encode_batch_size: int = 512,  # Q2 초기화용 h(a)를 추출할 때 Encoder batch size
    kmeans_n_init: int = 10,  # K-means를 다른 초기 centroid로 반복하는 횟수
    gumbel_t: float = 0.2,  # Gumbel-Softmax 사용 시 temperature
    split_batches: bool = True,  # multi-device 사용 시 batch 분할 여부
    amp: bool = True,  # mixed precision 사용 여부
    mixed_precision_type: str = "fp16",  # mixed precision dtype
    num_workers: int = 0,  # DataLoader worker 수
    do_eval: bool = False,  # 학습 중 validation 수행 여부
    eval_every: int = 5000,  # 몇 iteration마다 validation할지
    pretrained_rqvae_path=None,  # 이어서 학습할 checkpoint 경로
    save_dir_root: str = "out/rqvae/ebnerd",  # checkpoint 저장 폴더
    save_model_every: int = 5000,  # checkpoint 저장 주기
    wandb_logging: bool = False,  # W&B logging 사용 여부
    wandb_project: str = "news-rqvae-training",  # W&B project 이름
    seed: int = 42,  # random seed
):
    ensure_torch_serialization_compatibility()  # checkpoint 관련 torch 문제 확인
    set_seed(seed)  # 실험 재현성을 위한 seed 설정

    accelerator = Accelerator(
        split_batches=split_batches,  # 여러 device 사용 시 batch 처리 방식
        mixed_precision=mixed_precision_type if amp else "no",  # AMP 사용 시 fp16 등 적용
    )

    device = accelerator.device  # 현재 학습 device
    print(f"Device: {device}")

    os.makedirs(save_dir_root, exist_ok=True)  # checkpoint 저장 폴더 생성

    if accelerator.is_main_process:
        test_torch_serialization(save_dir_root)  # 학습 전 checkpoint 저장/load 테스트

    accelerator.wait_for_everyone()  # 여러 process가 모두 준비될 때까지 대기

    print_train_config(
        iterations=iterations,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulate_every=gradient_accumulate_every,
        dataset_folder=dataset_folder,
        vae_input_dim=vae_input_dim,
        vae_hidden_dims=vae_hidden_dims,
        vae_embed_dim=vae_embed_dim,
        vae_num_categories=vae_num_categories,
        vae_c2_codebook_size=vae_c2_codebook_size,
        vae_c3_codebook_size=vae_c3_codebook_size,
        vae_codebook_normalize=vae_codebook_normalize,
        vae_sim_vq=vae_sim_vq,
        vae_codebook_mode=vae_codebook_mode,
        lambda_rec=lambda_rec,
        lambda_cb=lambda_cb,
        lambda_com=lambda_com,
        amp=amp,
        mixed_precision_type=mixed_precision_type,
        save_dir_root=save_dir_root,
        do_eval=do_eval,
        seed=seed,
    )  # 현재 실제 학습 설정 출력

    train_dataset = NewsArticleDataset(
        data_dir=dataset_folder,
        split="train",
    )  # Train 기사 dataset 생성

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # 실제 학습 batch size
        shuffle=True,  # iteration마다 기사 순서를 섞음
        drop_last=False,  # 마지막 작은 batch도 사용
        num_workers=num_workers,
        pin_memory=device.type == "cuda",  # GPU 사용 시 CPU→GPU 전송 최적화
    )

    print(f"Train articles: {len(train_dataset)}")

    if do_eval:
        eval_dataset = NewsArticleDataset(
            data_dir=dataset_folder,
            split="validation",
        )  # validation dataset 생성

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,  # validation은 shuffle 불필요
            drop_last=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        print(f"Validation articles: {len(eval_dataset)}")
    else:
        eval_dataset = None  # validation 사용 안 함
        eval_dataloader = None

    first_item = train_dataset[0]  # dataset 구조 확인용 첫 기사
    first_x = first_item["x"]  # 첫 기사 embedding
    actual_input_dim = first_x.shape[-1]  # 실제 embedding 차원

    if actual_input_dim != vae_input_dim:
        raise ValueError(
            "Input dimension mismatch. "
            f"Dataset x(a) dimension={actual_input_dim}, "
            f"vae_input_dim={vae_input_dim}"
        )  # Gin의 입력 차원과 실제 데이터 차원이 다른 경우 오류

    if "category_id" not in first_item and "model_category_id" not in first_item:
        raise KeyError(
            "Dataset item must contain 'category_id' or 'model_category_id'."
        )  # Q1 deterministic ID에 필요한 category 정보 확인

    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        num_categories=vae_num_categories,
        c2_codebook_size=vae_c2_codebook_size,
        c3_codebook_size=vae_c3_codebook_size,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        lambda_rec=lambda_rec,
        lambda_cb=lambda_cb,
        lambda_com=lambda_com,
    )  # Gin 설정을 이용해 RQ-VAE 모델 생성

    model = model.to(device)  # Encoder/Decoder/Q1/Q2/Q3를 GPU/CPU로 이동

    start_iter = 0  # 처음부터 학습하면 iteration 0부터 시작
    checkpoint_state = None  # checkpoint가 없으면 None

    if pretrained_rqvae_path is not None:
        checkpoint_state = safe_torch_load(
            pretrained_rqvae_path,
            map_location=device,
        )  # 기존 checkpoint load

        model.load_state_dict(checkpoint_state["model"])  # 기존 모델 parameter 복원
        start_iter = checkpoint_state["iter"] + 1  # 저장된 다음 iteration부터 이어서 학습

        print(f"Loaded checkpoint: {pretrained_rqvae_path}")

    else:
        initialize_c2_codebook(
            model=model,
            train_dataset=train_dataset,
            device=device,
            c2_codebook_size=vae_c2_codebook_size,
            encode_batch_size=kmeans_encode_batch_size,
            kmeans_n_init=kmeans_n_init,
            seed=seed,
        )  # fresh training일 때만 전체 Train event로 Q2 K-means 초기화

    optimizer = AdamW(
        params=model.parameters(),  # Encoder/Decoder/Q1/Q2/Q3 모두 optimizer 대상
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if checkpoint_state is not None and "optimizer" in checkpoint_state:
        optimizer.load_state_dict(checkpoint_state["optimizer"])  # resume 시 optimizer 상태도 복원

    if do_eval:
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
        )  # 모델/optimizer/train/validation loader를 Accelerate에 등록
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
        )  # validation 없이 학습 요소만 Accelerate에 등록

    if wandb_logging and accelerator.is_main_process:
        wandb.login()  # W&B 계정 인증

        wandb.init(
            project=wandb_project,
            config={
                "iterations": iterations,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "vae_input_dim": vae_input_dim,
                "vae_hidden_dims": vae_hidden_dims,
                "vae_embed_dim": vae_embed_dim,
                "num_categories": vae_num_categories,
                "c2_codebook_size": vae_c2_codebook_size,
                "c3_codebook_size": vae_c3_codebook_size,
                "codebook_mode": str(vae_codebook_mode),
                "lambda_rec": lambda_rec,
                "lambda_cb": lambda_cb,
                "lambda_com": lambda_com,
                "seed": seed,
            },
        )  # W&B run 생성 및 실험 설정 저장

        wandb.define_metric("iteration")  # 그래프 x축으로 사용할 iteration 정의
        wandb.define_metric("train/*", step_metric="iteration")  # train 그래프 x축을 iteration으로 통일
        wandb.define_metric("eval/*", step_metric="iteration")  # eval 그래프 x축을 iteration으로 통일
        wandb.define_metric("learning_rate", step_metric="iteration")  # learning rate도 동일한 x축 사용

    if accelerator.is_main_process:
        gin_config_path = os.path.join(save_dir_root, "operative_config.gin")  # 실제 적용된 Gin config 저장 경로

        with open(gin_config_path, "w", encoding="utf-8") as f:
            f.write(gin.operative_config_str())  # 실제 실행된 설정 저장

    train_iterator = iter(train_dataloader)  # DataLoader에서 batch를 순차적으로 가져올 iterator 생성

    pbar = tqdm(
        range(start_iter, iterations),  # resume 여부를 반영해 시작 iteration 결정
        initial=start_iter,
        total=iterations,
        disable=not accelerator.is_main_process,
    )

    for iteration in pbar:
        model.train()  # training mode 활성화

        accumulated_loss = 0.0  # gradient accumulation 동안 total loss 합
        accumulated_rec_loss = 0.0  # reconstruction loss 합
        accumulated_cb_loss = 0.0  # codebook loss 합
        accumulated_com_loss = 0.0  # commitment loss 합
        accumulated_rqvae_loss = 0.0  # quantization loss 합

        optimizer.zero_grad(set_to_none=True)  # 이전 iteration gradient 제거

        for _ in range(gradient_accumulate_every):
            try:
                batch = next(train_iterator)  # 다음 train batch 가져오기
            except StopIteration:
                train_iterator = iter(train_dataloader)  # 한 epoch이 끝나면 iterator 새로 생성
                batch = next(train_iterator)

            x, category_ids, _ = unpack_batch(batch)  # 학습에서는 event ID 사용하지 않음

            x = x.to(
                device=device,
                non_blocking=True,
            )  # 기사 embedding을 GPU로 이동

            category_ids = category_ids.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )  # category ID를 GPU로 이동

            with accelerator.autocast():
                model_output = model(
                    x=x,
                    category_ids=category_ids,
                    gumbel_t=gumbel_t,
                )  # RQ-VAE forward 및 loss 계산

                loss = model_output.loss / gradient_accumulate_every  # accumulation 횟수만큼 loss를 나눠 gradient 규모 유지

            accelerator.backward(loss)  # mixed precision을 고려한 backward 수행

            accumulated_loss += loss.detach().float().item()  # total loss logging 값 누적
            accumulated_rec_loss += (
                model_output.reconstruction_loss.detach().float().item()
                / gradient_accumulate_every
            )  # reconstruction loss 누적
            accumulated_cb_loss += (
                model_output.codebook_loss.detach().float().item()
                / gradient_accumulate_every
            )  # codebook loss 누적
            accumulated_com_loss += (
                model_output.commitment_loss.detach().float().item()
                / gradient_accumulate_every
            )  # commitment loss 누적
            accumulated_rqvae_loss += (
                model_output.rqvae_loss.detach().float().item()
                / gradient_accumulate_every
            )  # weighted quantization loss 누적

        optimizer.step()  # Encoder/Decoder/Q1/Q2/Q3 parameter를 한 번 업데이트

        pbar.set_description(
            f"loss: {accumulated_loss:.4f} | "
            f"rec: {accumulated_rec_loss:.4f} | "
            f"cb: {accumulated_cb_loss:.4f} | "
            f"com: {accumulated_com_loss:.4f}"
        )  # progress bar에 현재 loss 출력

        if wandb_logging and accelerator.is_main_process:
            wandb.log(
                {
                    "iteration": iteration + 1,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train/total_loss": accumulated_loss,
                    "train/reconstruction_loss": accumulated_rec_loss,
                    "train/codebook_loss": accumulated_cb_loss,
                    "train/commitment_loss": accumulated_com_loss,
                    "train/rqvae_loss": accumulated_rqvae_loss,
                    "train/p_unique_ids": model_output.p_unique_ids.detach().float().cpu().item(),
                }
            )  # 각 iteration의 학습 지표를 W&B에 기록

        if do_eval and (iteration + 1) % eval_every == 0:
            accelerator.wait_for_everyone()  # evaluation 전에 모든 process 동기화

            eval_result = evaluate(
                model=model,
                dataloader=eval_dataloader,
                device=device,
                gumbel_t=gumbel_t,
            )  # validation loss 계산

            if accelerator.is_main_process:
                print(
                    f"\n[Eval {iteration + 1}] "
                    f"loss={eval_result['loss']:.4f}, "
                    f"rec={eval_result['reconstruction_loss']:.4f}, "
                    f"cb={eval_result['codebook_loss']:.4f}, "
                    f"com={eval_result['commitment_loss']:.4f}"
                )

                if wandb_logging:
                    wandb.log(
                        {
                            "iteration": iteration + 1,
                            "eval/total_loss": eval_result["loss"],
                            "eval/reconstruction_loss": eval_result["reconstruction_loss"],
                            "eval/codebook_loss": eval_result["codebook_loss"],
                            "eval/commitment_loss": eval_result["commitment_loss"],
                            "eval/rqvae_loss": eval_result["rqvae_loss"],
                        }
                    )  # validation 결과도 W&B에 기록

        should_save = (iteration + 1) % save_model_every == 0  # checkpoint 저장 iteration인지 확인

        if should_save:
            accelerator.wait_for_everyone()  # 저장 전에 모든 process 동기화

            if accelerator.is_main_process:
                ensure_torch_serialization_compatibility()  # 저장 직전 torch 상태 확인

                state = build_checkpoint_state(
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                    lambda_rec=lambda_rec,
                    lambda_cb=lambda_cb,
                    lambda_com=lambda_com,
                )  # checkpoint에 저장할 전체 상태 생성

                checkpoint_path = os.path.join(
                    save_dir_root,
                    f"checkpoint_{iteration + 1}.pt",
                )  # checkpoint 파일 이름 생성

                safe_torch_save(state, checkpoint_path)  # checkpoint 안전하게 저장
                print(f"\nSaved checkpoint: {checkpoint_path}")

    accelerator.wait_for_everyone()  # 전체 학습 종료 후 process 동기화

    if accelerator.is_main_process:
        ensure_torch_serialization_compatibility()  # 최종 저장 전 torch 상태 확인

        final_iteration = iterations - 1  # 마지막 실제 iteration index

        final_state = build_checkpoint_state(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            iteration=final_iteration,
            lambda_rec=lambda_rec,
            lambda_cb=lambda_cb,
            lambda_com=lambda_com,
        )  # 최종 checkpoint 상태 생성

        final_path = os.path.join(save_dir_root, "checkpoint_final.pt")  # 최종 모델 저장 경로
        safe_torch_save(final_state, final_path)  # 최종 checkpoint 저장

        print("\n" + "=" * 70)
        print(f"Final model saved: {final_path}")
        print("=" * 70)

    if wandb_logging and accelerator.is_main_process:
        wandb.finish()  # W&B run 정상 종료


if __name__ == "__main__":
    parse_config()  # 실행 시 Gin config 파일 읽기
    train()  # 설정을 적용해 RQ-VAE 학습 시작