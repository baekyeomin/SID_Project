from __future__ import annotations

from typing import NamedTuple

import gin
import torch
import torch.nn.functional as F

from torch import Tensor, nn
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack


class TransformerModelOutput(NamedTuple):
    c1_logits: Tensor
    c2_logits: Tensor
    c3_logits: Tensor


class CandidateScoreOutput(NamedTuple):
    candidate_scores: Tensor
    c1_log_probs: Tensor
    c2_log_probs: Tensor
    c3_log_probs: Tensor
    tie_scores: Tensor


class EncoderOutput(NamedTuple):
    hidden_states: Tensor
    attention_mask: Tensor


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

        self.bos_embedding = nn.Parameter(
            torch.empty(
                1,
                1,
                d_model,
            )
        )

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

        self.encoder_dummy_embedding = nn.Embedding(
            1,
            d_model,
        )

        self.decoder_dummy_embedding = nn.Embedding(
            1,
            d_model,
        )

        self.encoder = T5Stack(encoder_config)
        self.encoder.set_input_embeddings(
            self.encoder_dummy_embedding
        )

        self.decoder = T5Stack(decoder_config)
        self.decoder.set_input_embeddings(
            self.decoder_dummy_embedding
        )

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

        self.history_tie_proj = nn.Linear(
            d_model,
            d_model,
        )

        self.candidate_tie_proj = nn.Sequential(
            nn.Linear(
                d_model * 4,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout_rate
            ),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        self.tie_logit_scale = nn.Parameter(
            torch.tensor(0.0)
        )

        self._reset_parameters()

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

        if history_sids.shape[-1] != 4:
            raise ValueError(
                "history_sids must contain c1,c2,c3,c4."
            )

        if history_mask.ndim != 2:
            raise ValueError(
                "history_mask must have shape [B,H]."
            )

        if history_sids.shape[:2] != history_mask.shape:
            raise ValueError(
                "history_sids and history_mask dimensions must match."
            )

    def _validate_candidates(
        self,
        candidate_sids: Tensor,
        candidate_c4: Tensor,
    ) -> None:
        if candidate_sids.ndim != 3:
            raise ValueError(
                "candidate_sids must have shape [B,C,3]. "
                f"Received: {tuple(candidate_sids.shape)}"
            )

        if candidate_sids.shape[-1] != 3:
            raise ValueError(
                "candidate_sids must contain exactly c1,c2,c3."
            )

        if candidate_c4.ndim != 2:
            raise ValueError(
                "candidate_c4 must have shape [B,C]. "
                f"Received: {tuple(candidate_c4.shape)}"
            )

        if candidate_sids.shape[:2] != candidate_c4.shape:
            raise ValueError(
                "candidate_sids [B,C] and candidate_c4 [B,C] "
                "dimensions must match."
            )

    def embed_history(
        self,
        history_sids: Tensor,
    ) -> Tensor:
        c1 = history_sids[:, :, 0]
        c2 = history_sids[:, :, 1]
        c3 = history_sids[:, :, 2]
        c4 = history_sids[:, :, 3]

        article_embeddings = torch.stack(
            [
                self.c1_embedding(c1),
                self.c2_embedding(c2),
                self.c3_embedding(c3),
                self.c4_embedding(c4),
            ],
            dim=2,
        )

        if self.use_sep:
            batch_size = history_sids.shape[0]
            history_length = history_sids.shape[1]

            sep = self.sep_embedding.expand(
                batch_size,
                history_length,
                -1,
            ).unsqueeze(2)

            article_embeddings = torch.cat(
                [
                    article_embeddings,
                    sep,
                ],
                dim=2,
            )

        return article_embeddings.reshape(
            article_embeddings.shape[0],
            -1,
            self.d_model,
        )

    def expand_history_mask(
        self,
        history_mask: Tensor,
    ) -> Tensor:
        tokens_per_article = (
            5 if self.use_sep else 4
        )

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

    def encode(
        self,
        history_sids: Tensor,
        history_mask: Tensor,
    ) -> EncoderOutput:
        self._validate_history(
            history_sids,
            history_mask,
        )

        encoder_inputs = self.embed_history(
            history_sids
        )

        encoder_attention_mask = (
            self.expand_history_mask(
                history_mask
            ).long()
        )

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

    def pool_history(
        self,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        mask = (
            encoder_attention_mask
            .unsqueeze(-1)
            .to(
                encoder_hidden_states.dtype
            )
        )

        summed = (
            encoder_hidden_states
            * mask
        ).sum(
            dim=1
        )

        denominator = (
            mask.sum(
                dim=1
            )
            .clamp_min(
                1.0
            )
        )

        return (
            summed
            / denominator
        )

    def build_teacher_forcing_inputs(
        self,
        sids: Tensor,
    ) -> Tensor:
        if (
            sids.ndim != 2
            or sids.shape[-1] != 3
        ):
            raise ValueError(
                "sids must have shape [N,3]."
            )

        batch_size = sids.shape[0]

        bos = self.bos_embedding.expand(
            batch_size,
            -1,
            -1,
        )

        c1_emb = self.c1_embedding(
            sids[:, 0]
        ).unsqueeze(1)

        c2_emb = self.c2_embedding(
            sids[:, 1]
        ).unsqueeze(1)

        return torch.cat(
            [
                bos,
                c1_emb,
                c2_emb,
            ],
            dim=1,
        )

    def decode(
        self,
        decoder_inputs: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        decoder_outputs = self.decoder(
            inputs_embeds=
                decoder_inputs,

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

    def _get_sid_logits(
        self,
        sids: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> TransformerModelOutput:
        decoder_inputs = (
            self.build_teacher_forcing_inputs(
                sids
            )
        )

        decoder_hidden = self.decode(
            decoder_inputs=
                decoder_inputs,

            encoder_hidden_states=
                encoder_hidden_states,

            encoder_attention_mask=
                encoder_attention_mask,
        )

        c1_logits = self.c1_head(
            decoder_hidden[
                :,
                0,
                :,
            ]
        )

        c2_logits = self.c2_head(
            decoder_hidden[
                :,
                1,
                :,
            ]
        )

        c3_logits = self.c3_head(
            decoder_hidden[
                :,
                2,
                :,
            ]
        )

        return TransformerModelOutput(
            c1_logits=
                c1_logits,

            c2_logits=
                c2_logits,

            c3_logits=
                c3_logits,
        )

    def build_candidate_tie_vectors(
        self,
        candidate_sids: Tensor,
        candidate_c4: Tensor,
    ) -> Tensor:
        c1_emb = self.c1_embedding(
            candidate_sids[
                :,
                :,
                0,
            ]
        )

        c2_emb = self.c2_embedding(
            candidate_sids[
                :,
                :,
                1,
            ]
        )

        c3_emb = self.c3_embedding(
            candidate_sids[
                :,
                :,
                2,
            ]
        )

        c4_emb = self.c4_embedding(
            candidate_c4
        )

        candidate_features = torch.cat(
            [
                c1_emb,
                c2_emb,
                c3_emb,
                c4_emb,
            ],
            dim=-1,
        )

        return self.candidate_tie_proj(
            candidate_features
        )

    def compute_tie_scores(
        self,
        encoder_output: EncoderOutput,
        candidate_sids: Tensor,
        candidate_c4: Tensor,
    ) -> Tensor:
        history_vector = self.pool_history(
            encoder_hidden_states=
                encoder_output.hidden_states,

            encoder_attention_mask=
                encoder_output.attention_mask,
        )

        history_vector = self.history_tie_proj(
            history_vector
        )

        candidate_vectors = (
            self.build_candidate_tie_vectors(
                candidate_sids=
                    candidate_sids,

                candidate_c4=
                    candidate_c4,
            )
        )

        history_vector = F.normalize(
            history_vector,
            dim=-1,
        )

        candidate_vectors = F.normalize(
            candidate_vectors,
            dim=-1,
        )

        scale = (
            self.tie_logit_scale
            .exp()
            .clamp(
                max=100.0
            )
        )

        return (
            scale
            * (
                history_vector.unsqueeze(1)
                * candidate_vectors
            ).sum(
                dim=-1
            )
        )

    def score_candidates(
        self,
        history_sids: Tensor,
        history_mask: Tensor,
        candidate_sids: Tensor,
        candidate_c4: Tensor,
    ) -> CandidateScoreOutput:
        self._validate_candidates(
            candidate_sids=
                candidate_sids,

            candidate_c4=
                candidate_c4,
        )

        if (
            history_sids.shape[0]
            != candidate_sids.shape[0]
        ):
            raise ValueError(
                "history_sids and candidate_sids "
                "batch sizes must match."
            )

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

        flat_candidate_sids = (
            candidate_sids.reshape(
                batch_size
                * num_candidates,
                3,
            )
        )

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

        candidate_scores = (
            c1_log_probs
            + c2_log_probs
            + c3_log_probs
        )

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

        tie_scores = self.compute_tie_scores(
            encoder_output=
                encoder_output,

            candidate_sids=
                candidate_sids,

            candidate_c4=
                candidate_c4,
        )

        return CandidateScoreOutput(
            candidate_scores=
                candidate_scores,

            c1_log_probs=
                c1_log_probs,

            c2_log_probs=
                c2_log_probs,

            c3_log_probs=
                c3_log_probs,

            tie_scores=
                tie_scores,
        )

    def forward(
        self,
        history_sids: Tensor,
        history_mask: Tensor,
        candidate_sids: Tensor,
        candidate_c4: Tensor,
    ) -> CandidateScoreOutput:
        return self.score_candidates(
            history_sids=
                history_sids,

            history_mask=
                history_mask,

            candidate_sids=
                candidate_sids,

            candidate_c4=
                candidate_c4,
        )


if __name__ == "__main__":
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

    candidate_sids = torch.tensor(
        [
            [
                [2, 6, 7],
                [2, 6, 7],
                [1, 8, 5],
            ],
            [
                [3, 4, 9],
                [3, 4, 9],
                [0, 0, 0],
            ],
        ],
        dtype=torch.long,
    )

    candidate_c4 = torch.tensor(
        [
            [
                0,
                1,
                0,
            ],
            [
                0,
                1,
                0,
            ],
        ],
        dtype=torch.long,
    )

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

    output = model(
        history_sids=
            history_sids,

        history_mask=
            history_mask,

        candidate_sids=
            candidate_sids,

        candidate_c4=
            candidate_c4,
    )

    print(
        "candidate_scores:",
        output.candidate_scores.shape,
    )

    print(
        output.candidate_scores
    )

    print(
        "c1_log_probs:",
        output.c1_log_probs.shape,
    )

    print(
        "c2_log_probs:",
        output.c2_log_probs.shape,
    )

    print(
        "c3_log_probs:",
        output.c3_log_probs.shape,
    )

    print(
        "tie_scores:",
        output.tie_scores.shape,
    )

    print(
        output.tie_scores
    )