
## 1. EB-NeRD — H 버전

### ① RQ-VAE 모델 학습

```bash
python train_rqvae_hver.py --config configs/rqvae_ebnerd_hver.gin
```

체크포인트 저장: `out/rqvae/ebnerd_h/checkpoint_final.pt`

### ② SID 생성

```python
from generate_semantic_ids_timeupdate_hver import generate_semantic_ids

generate_semantic_ids(
    data_dir="datasets/ebnerd",
    checkpoint_path="out/rqvae/ebnerd_h/checkpoint_final.pt",
    output_dir="out/semantic_ids/ebnerd_h",
    batch_size=512,
)
```

---

## 2. MIND — H 버전

### ① RQ-VAE 모델 학습

```bash
python train_rqvae_hver.py --config configs/rqvae_mind_hver.gin
```

체크포인트 저장: `out/rqvae/mind_h/checkpoint_final.pt`

### ② SID 생성

```python
from generate_semantic_ids_timeupdate_hver import generate_semantic_ids

generate_semantic_ids(
    data_dir="datasets/mind",
    checkpoint_path="out/rqvae/mind_h/checkpoint_final.pt",
    output_dir="out/semantic_ids/mind_h",
    batch_size=512,
)
```

---

## 3. 결과 저장 경로

| 데이터셋 | 모델 체크포인트 | SID 결과 |
|---|---|---|
| EB-NeRD H | `out/rqvae/ebnerd_h/` | `out/semantic_ids/ebnerd_h/` |
| MIND H | `out/rqvae/mind_h/` | `out/semantic_ids/mind_h/` |
