from __future__ import annotations

from typing import NamedTuple

import gin
import torch

from torch import Tensor, nn
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack


# ============================================================
# Constants
# ============================================================

NUM_HISTORY_SID_LEVELS = 4      # c1, c2, c3, c4
NUM_CANDIDATE_SID_LEVELS = 3    # c1, c2, c3
NUM_CANDIDATES = 5              # positive 1 + negative 4


# ============================================================
# Output structures
# ============================================================

class TransformerModelOutput(NamedTuple):
    """
    Decoder가 각 SID level에 대해 출력한 logits.
    """
    c1_logits: Tensor
    c2_logits: Tensor
    c3_logits: Tensor


class CandidateScoreOutput(NamedTuple):
    """
    각 candidate의 최종 score와
    c1/c2/c3 level별 log probability.
    """
    candidate_scores: Tensor
    c1_log_probs: Tensor
    c2_log_probs: Tensor
    c3_log_probs: Tensor


class EncoderOutput(NamedTuple):
    """
    History encoder의 출력.
    """
    hidden_states: Tensor
    attention_mask: Tensor


# ============================================================
# Transformer
# ============================================================

@gin.configurable
class NewsEncoderDecoderTransformer(nn.Module):

    def __init__(
        self,
        c1_vocab_size: int,
        c2_vocab_size: int,
        c3_vocab_size: int,
        c4_vocab_size: int,
        d_model: int = 384,
        num_heads: int = 6,
        d_ff: int = 1024,
        num_layers: int = 4,
        dropout_rate: float = 0.1,
        use_sep: bool = True,
    ) -> None:

        super().__init__()

        # ====================================================
        # 기본 parameter 검증
        # ====================================================

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        if min(
            c1_vocab_size,
            c2_vocab_size,
            c3_vocab_size,
            c4_vocab_size,
        ) <= 0:
            raise ValueError(
                "All vocabulary sizes must be > 0."
            )

        # ====================================================
        # Hyperparameters 저장
        # ====================================================

        self.c1_vocab_size = c1_vocab_size
        self.c2_vocab_size = c2_vocab_size
        self.c3_vocab_size = c3_vocab_size
        self.c4_vocab_size = c4_vocab_size

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_sep = use_sep

        # ====================================================
        # SID embeddings
        #
        # history에서는 c1,c2,c3,c4 모두 사용
        # candidate에서는 c1,c2,c3만 사용
        # ====================================================

        self.c1_embedding = nn.Embedding(
            c1_vocab_size,
            d_model,
        )

        self.c2_embedding = nn.Embedding(
            c2_vocab_size,
            d_model,
        )

        self.c3_embedding = nn.Embedding(
            c3_vocab_size,
            d_model,
        )

        self.c4_embedding = nn.Embedding(
            c4_vocab_size,
            d_model,
        )

        # ====================================================
        # Decoder BOS embedding
        #
        # decoder input:
        # [BOS, c1, c2]
        #
        # 각각의 위치에서
        # c1, c2, c3를 예측
        # ====================================================

        self.bos_embedding = nn.Parameter(
            torch.empty(
                1,
                1,
                d_model,
            )
        )

        # ====================================================
        # History 기사 사이를 구분하는 SEP
        # ====================================================

        if use_sep:

            self.sep_embedding = nn.Parameter(
                torch.empty(
                    1,
                    1,
                    d_model,
                )
            )

        else:

            self.register_parameter(
                "sep_embedding",
                None,
            )

        # ====================================================
        # T5 Encoder config
        # ====================================================

        encoder_config = T5Config(
            vocab_size=1,
            d_model=d_model,
            d_kv=d_model // num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            num_decoder_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            is_decoder=False,
            is_encoder_decoder=False,
            use_cache=False,
            eos_token_id=None,
        )

        # ====================================================
        # T5 Decoder config
        # ====================================================

        decoder_config = T5Config(
            vocab_size=1,
            d_model=d_model,
            d_kv=d_model // num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            num_decoder_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            is_decoder=True,
            is_encoder_decoder=False,
            use_cache=False,
            eos_token_id=None,
        )

        # ====================================================
        # Dummy embeddings
        #
        # 실제 input은 inputs_embeds로 직접 전달하므로
        # vocabulary embedding은 사용하지 않는다.
        # ====================================================

        self.encoder_dummy_embedding = nn.Embedding(
            1,
            d_model,
        )

        self.decoder_dummy_embedding = nn.Embedding(
            1,
            d_model,
        )

        # ====================================================
        # T5 Encoder
        # ====================================================

        self.encoder = T5Stack(
            encoder_config
        )

        self.encoder.set_input_embeddings(
            self.encoder_dummy_embedding
        )

        # ====================================================
        # T5 Decoder
        # ====================================================

        self.decoder = T5Stack(
            decoder_config
        )

        self.decoder.set_input_embeddings(
            self.decoder_dummy_embedding
        )

        # ====================================================
        # SID prediction heads
        # ====================================================

        self.c1_head = nn.Linear(
            d_model,
            c1_vocab_size,
        )

        self.c2_head = nn.Linear(
            d_model,
            c2_vocab_size,
        )

        self.c3_head = nn.Linear(
            d_model,
            c3_vocab_size,
        )

        # ====================================================
        # Parameter 초기화
        # ====================================================

        self._reset_parameters()

    # ========================================================
    # Parameter initialization
    # ========================================================

    def _reset_parameters(self) -> None:

        std = self.d_model ** -0.5

        nn.init.normal_(
            self.c1_embedding.weight,
            mean=0.0,
            std=std,
        )

        nn.init.normal_(
            self.c2_embedding.weight,
            mean=0.0,
            std=std,
        )

        nn.init.normal_(
            self.c3_embedding.weight,
            mean=0.0,
            std=std,
        )

        nn.init.normal_(
            self.c4_embedding.weight,
            mean=0.0,
            std=std,
        )

        nn.init.normal_(
            self.bos_embedding,
            mean=0.0,
            std=std,
        )

        if self.sep_embedding is not None:

            nn.init.normal_(
                self.sep_embedding,
                mean=0.0,
                std=std,
            )

    # ========================================================
    # History shape 검증
    # ========================================================

    def _validate_history(
        self,
        history_sids: Tensor,
        history_mask: Tensor,
    ) -> None:

        if history_sids.ndim != 3:

            raise ValueError(
                "history_sids must have shape [B,H,4]. "
                f"Received: {tuple(history_sids.shape)}"
            )

        if (
            history_sids.shape[-1]
            != NUM_HISTORY_SID_LEVELS
        ):

            raise ValueError(
                "history_sids must contain "
                "c1,c2,c3,c4."
            )

        if history_mask.ndim != 2:

            raise ValueError(
                "history_mask must have shape [B,H]."
            )

        if (
            history_sids.shape[:2]
            != history_mask.shape
        ):

            raise ValueError(
                "history_sids and history_mask "
                "dimensions must match."
            )

    # ========================================================
    # Candidate shape 검증
    # ========================================================

    def _validate_candidates(
        self,
        candidate_sids: Tensor,
    ) -> None:

        if candidate_sids.ndim != 3:

            raise ValueError(
                "candidate_sids must have shape [B,5,3]. "
                f"Received: {tuple(candidate_sids.shape)}"
            )

        if (
            candidate_sids.shape[-1]
            != NUM_CANDIDATE_SID_LEVELS
        ):

            raise ValueError(
                "candidate_sids must contain "
                "exactly c1,c2,c3."
            )

        if (
            candidate_sids.shape[1]
            != NUM_CANDIDATES
        ):

            raise ValueError(
                f"Expected {NUM_CANDIDATES} candidates, "
                f"but received "
                f"{candidate_sids.shape[1]}."
            )

    # ========================================================
    # History SID embedding
    # ========================================================

    def embed_history(
        self,
        history_sids: Tensor,
    ) -> Tensor:

        # [B, H]
        c1 = history_sids[:, :, 0]
        c2 = history_sids[:, :, 1]
        c3 = history_sids[:, :, 2]
        c4 = history_sids[:, :, 3]

        # 각 SID level embedding
        # 각각 [B,H,d_model]
        c1_emb = self.c1_embedding(c1)
        c2_emb = self.c2_embedding(c2)
        c3_emb = self.c3_embedding(c3)
        c4_emb = self.c4_embedding(c4)

        # [B,H,4,d_model]
        article_embeddings = torch.stack(
            [
                c1_emb,
                c2_emb,
                c3_emb,
                c4_emb,
            ],
            dim=2,
        )

        # ====================================================
        # SEP token 추가
        #
        # 한 기사:
        # c1 c2 c3 c4 SEP
        # ====================================================

        if self.use_sep:

            batch_size = (
                history_sids.shape[0]
            )

            history_length = (
                history_sids.shape[1]
            )

            sep = (
                self.sep_embedding
                .expand(
                    batch_size,
                    history_length,
                    -1,
                )
                .unsqueeze(2)
            )

            article_embeddings = torch.cat(
                [
                    article_embeddings,
                    sep,
                ],
                dim=2,
            )

        # ----------------------------------------------------
        # [B,H,5,D]
        #       ↓
        # [B,H*5,D]
        #
        # use_sep=False라면 [B,H*4,D]
        # ----------------------------------------------------

        return article_embeddings.reshape(
            article_embeddings.shape[0],
            -1,
            self.d_model,
        )

    # ========================================================
    # History mask 확장
    # ========================================================

    def expand_history_mask(
        self,
        history_mask: Tensor,
    ) -> Tensor:

        tokens_per_article = (
            5 if self.use_sep else 4
        )

        # [B,H]
        #   ↓
        # [B,H,tokens_per_article]
        #   ↓
        # [B,H*tokens_per_article]

        return (
            history_mask
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                tokens_per_article,
            )
            .reshape(
                history_mask.shape[0],
                -1,
            )
        )

    # ========================================================
    # Encoder
    # ========================================================

    def encode(
        self,
        history_sids: Tensor,
        history_mask: Tensor,
    ) -> EncoderOutput:

        self._validate_history(
            history_sids,
            history_mask,
        )

        # SID → embedding
        encoder_inputs = self.embed_history(
            history_sids
        )

        # article mask → token mask
        encoder_attention_mask = (
            self.expand_history_mask(
                history_mask
            ).long()
        )

        # T5 Encoder
        encoder_outputs = self.encoder(
            inputs_embeds=encoder_inputs,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        return EncoderOutput(
            hidden_states=
                encoder_outputs.last_hidden_state,

            attention_mask=
                encoder_attention_mask,
        )

    # ========================================================
    # Teacher forcing decoder input 생성
    #
    # candidate SID = [c1,c2,c3]
    #
    # decoder input =
    # [BOS, c1, c2]
    #
    # position 0 → c1 예측
    # position 1 → c2 예측
    # position 2 → c3 예측
    # ========================================================

    def build_teacher_forcing_inputs(
        self,
        sids: Tensor,
    ) -> Tensor:

        if (
            sids.ndim != 2
            or sids.shape[-1]
            != NUM_CANDIDATE_SID_LEVELS
        ):

            raise ValueError(
                "sids must have shape [N,3]."
            )

        batch_size = sids.shape[0]

        # [N,1,D]
        bos = self.bos_embedding.expand(
            batch_size,
            -1,
            -1,
        )

        # [N,1,D]
        c1_emb = self.c1_embedding(
            sids[:, 0]
        ).unsqueeze(1)

        # [N,1,D]
        c2_emb = self.c2_embedding(
            sids[:, 1]
        ).unsqueeze(1)

        # [N,3,D]
        return torch.cat(
            [
                bos,
                c1_emb,
                c2_emb,
            ],
            dim=1,
        )

    # ========================================================
    # Decoder
    # ========================================================

    def decode(
        self,
        decoder_inputs: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs,

            encoder_hidden_states=
                encoder_hidden_states,

            encoder_attention_mask=
                encoder_attention_mask,

            use_cache=False,
            return_dict=True,
        )

        return (
            decoder_outputs
            .last_hidden_state
        )

    # ========================================================
    # c1/c2/c3 logits 계산
    # ========================================================

    def _get_sid_logits(
        self,
        sids: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> TransformerModelOutput:

        # ----------------------------------------------------
        # [BOS, c1, c2]
        # ----------------------------------------------------

        decoder_inputs = (
            self.build_teacher_forcing_inputs(
                sids
            )
        )

        # ----------------------------------------------------
        # history와 cross-attention하며
        # candidate SID autoregressive decoding
        # ----------------------------------------------------

        decoder_hidden = self.decode(
            decoder_inputs=
                decoder_inputs,

            encoder_hidden_states=
                encoder_hidden_states,

            encoder_attention_mask=
                encoder_attention_mask,
        )

        # ----------------------------------------------------
        # Decoder position 0:
        # p(c1 | H)
        # ----------------------------------------------------

        c1_logits = self.c1_head(
            decoder_hidden[
                :,
                0,
                :,
            ]
        )

        # ----------------------------------------------------
        # Decoder position 1:
        # p(c2 | H,c1)
        # ----------------------------------------------------

        c2_logits = self.c2_head(
            decoder_hidden[
                :,
                1,
                :,
            ]
        )

        # ----------------------------------------------------
        # Decoder position 2:
        # p(c3 | H,c1,c2)
        # ----------------------------------------------------

        c3_logits = self.c3_head(
            decoder_hidden[
                :,
                2,
                :,
            ]
        )

        return TransformerModelOutput(
            c1_logits=c1_logits,
            c2_logits=c2_logits,
            c3_logits=c3_logits,
        )

    # ========================================================
    # Candidate scoring
    # ========================================================

    def score_candidates(
        self,
        history_sids: Tensor,
        history_mask: Tensor,
        candidate_sids: Tensor,
    ) -> CandidateScoreOutput:

        # ----------------------------------------------------
        # Candidate 입력 검증
        # ----------------------------------------------------

        self._validate_candidates(
            candidate_sids
        )

        if (
            history_sids.shape[0]
            != candidate_sids.shape[0]
        ):

            raise ValueError(
                "history_sids and candidate_sids "
                "batch sizes must match."
            )

        # ====================================================
        # 1. History를 한 번만 Encoder 통과
        # ====================================================

        encoder_output = self.encode(
            history_sids=
                history_sids,

            history_mask=
                history_mask,
        )

        batch_size = (
            candidate_sids.shape[0]
        )

        num_candidates = (
            candidate_sids.shape[1]
        )

        encoder_seq_len = (
            encoder_output
            .hidden_states
            .shape[1]
        )

        # ====================================================
        # 2. Candidate 5개를 하나의 batch처럼 flatten
        #
        # [B,5,3]
        #    ↓
        # [B*5,3]
        # ====================================================

        flat_candidate_sids = (
            candidate_sids.reshape(
                batch_size
                * num_candidates,
                NUM_CANDIDATE_SID_LEVELS,
            )
        )

        # ====================================================
        # 3. 같은 history를 candidate 5개 각각에 반복
        #
        # history 하나당 candidate가 5개이므로
        # Encoder output을 5번 복제
        # ====================================================

        repeated_encoder_hidden = (
            encoder_output
            .hidden_states
            .unsqueeze(1)
            .expand(
                -1,
                num_candidates,
                -1,
                -1,
            )
            .reshape(
                batch_size
                * num_candidates,
                encoder_seq_len,
                self.d_model,
            )
        )

        repeated_encoder_mask = (
            encoder_output
            .attention_mask
            .unsqueeze(1)
            .expand(
                -1,
                num_candidates,
                -1,
            )
            .reshape(
                batch_size
                * num_candidates,
                encoder_seq_len,
            )
        )

        # ====================================================
        # 4. 각 candidate의 c1/c2/c3 logits
        # ====================================================

        model_output = (
            self._get_sid_logits(
                sids=
                    flat_candidate_sids,

                encoder_hidden_states=
                    repeated_encoder_hidden,

                encoder_attention_mask=
                    repeated_encoder_mask,
            )
        )

        # ====================================================
        # 5. logits → log probability
        # ====================================================

        c1_all_log_probs = (
            torch.log_softmax(
                model_output.c1_logits,
                dim=-1,
            )
        )

        c2_all_log_probs = (
            torch.log_softmax(
                model_output.c2_logits,
                dim=-1,
            )
        )

        c3_all_log_probs = (
            torch.log_softmax(
                model_output.c3_logits,
                dim=-1,
            )
        )

        # ====================================================
        # 6. 실제 candidate가 가진 SID의 probability만 선택
        # ====================================================

        c1_log_probs = (
            c1_all_log_probs
            .gather(
                dim=1,

                index=
                    flat_candidate_sids[
                        :,
                        0,
                    ]
                    .unsqueeze(1),
            )
            .squeeze(1)
        )

        c2_log_probs = (
            c2_all_log_probs
            .gather(
                dim=1,

                index=
                    flat_candidate_sids[
                        :,
                        1,
                    ]
                    .unsqueeze(1),
            )
            .squeeze(1)
        )

        c3_log_probs = (
            c3_all_log_probs
            .gather(
                dim=1,

                index=
                    flat_candidate_sids[
                        :,
                        2,
                    ]
                    .unsqueeze(1),
            )
            .squeeze(1)
        )

        # ====================================================
        # 7. Candidate main score
        #
        # S(a|H)
        # =
        # log p(c1|H)
        # + log p(c2|H,c1)
        # + log p(c3|H,c1,c2)
        # ====================================================

        candidate_scores = (
            c1_log_probs
            + c2_log_probs
            + c3_log_probs
        )

        # ====================================================
        # 8. [B*5] → [B,5]
        # ====================================================

        candidate_scores = (
            candidate_scores.reshape(
                batch_size,
                num_candidates,
            )
        )

        c1_log_probs = (
            c1_log_probs.reshape(
                batch_size,
                num_candidates,
            )
        )

        c2_log_probs = (
            c2_log_probs.reshape(
                batch_size,
                num_candidates,
            )
        )

        c3_log_probs = (
            c3_log_probs.reshape(
                batch_size,
                num_candidates,
            )
        )

        # ====================================================
        # 출력
        # ====================================================

        return CandidateScoreOutput(
            candidate_scores=
                candidate_scores,

            c1_log_probs=
                c1_log_probs,

            c2_log_probs=
                c2_log_probs,

            c3_log_probs=
                c3_log_probs,
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        history_sids: Tensor,
        history_mask: Tensor,
        candidate_sids: Tensor,
    ) -> CandidateScoreOutput:

        return self.score_candidates(
            history_sids=
                history_sids,

            history_mask=
                history_mask,

            candidate_sids=
                candidate_sids,
        )


# ============================================================
# Simple test
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Example history
    #
    # B = 2
    # H = 3
    # SID = c1,c2,c3,c4
    # ========================================================

    history_sids = torch.tensor(
        [
            [
                [1, 2, 3, 0],
                [1, 3, 2, 0],
                [1, 5, 5, 1],
            ],
            [
                [2, 1, 4, 0],
                [3, 7, 8, 0],
                [0, 0, 0, 0],
            ],
        ],
        dtype=torch.long,
    )

    history_mask = torch.tensor(
        [
            [
                True,
                True,
                True,
            ],
            [
                True,
                True,
                False,
            ],
        ],
        dtype=torch.bool,
    )

    # ========================================================
    # Example candidates
    #
    # B = 2
    # candidate = 5개
    # candidate SID = c1,c2,c3
    #
    # 같은 row 안에서는 c1,c2,c3가 서로 다름
    # ========================================================

    candidate_sids = torch.tensor(
        [
            [
                [2, 6, 7],
                [2, 6, 8],
                [1, 8, 5],
                [3, 9, 4],
                [4, 10, 6],
            ],
            [
                [3, 4, 9],
                [3, 4, 10],
                [2, 8, 7],
                [4, 3, 6],
                [1, 5, 11],
            ],
        ],
        dtype=torch.long,
    )

    # ========================================================
    # Model
    # ========================================================

    model = NewsEncoderDecoderTransformer(
        c1_vocab_size=25,
        c2_vocab_size=128,
        c3_vocab_size=512,
        c4_vocab_size=28,
        d_model=384,
        num_heads=6,
        d_ff=1024,
        num_layers=4,
        dropout_rate=0.1,
        use_sep=True,
    )

    # ========================================================
    # Forward
    # ========================================================

    output = model(
        history_sids=
            history_sids,

        history_mask=
            history_mask,

        candidate_sids=
            candidate_sids,
    )

    # ========================================================
    # 결과 확인
    # ========================================================

    print(
        "candidate_scores:",
        output.candidate_scores.shape,
    )

    print(
        output.candidate_scores
    )

    print()

    print(
        "c1_log_probs:",
        output.c1_log_probs.shape,
    )

    print(
        output.c1_log_probs
    )

    print()

    print(
        "c2_log_probs:",
        output.c2_log_probs.shape,
    )

    print(
        output.c2_log_probs
    )

    print()

    print(
        "c3_log_probs:",
        output.c3_log_probs.shape,
    )

    print(
        output.c3_log_probs
    )