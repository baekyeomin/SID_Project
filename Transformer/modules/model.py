from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import gin
import torch

from torch import Tensor, nn
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack

# Model Output
class TransformerModelOutput(NamedTuple):
    """
    Transformer가 각 SID hierarchy에 대해 출력한 logits
    [Shapes]------
    c1_logits : [B, c1_vocab_size]
    c2_logits : [B, c2_vocab_size]
    c3_logits : [B, c3_vocab_size]
    c4_logits : [B, c4_vocab_size]
    """
    c1_logits: Tensor
    c2_logits: Tensor
    c3_logits: Tensor
    c4_logits: Tensor


# Encoder Output
class EncoderOutput(NamedTuple):
    """
    predict_sid.py에서도 encoder 결과를 재사용하기 위한 구조

    hidden_states: [B, encoder_seq_len, d_model]
    attention_mask: [B, encoder_seq_len]
    """
    hidden_states: Tensor
    attention_mask: Tensor


# Encoder-Decoder Transformer
@gin.configurable
class NewsEncoderDecoderTransformer(nn.Module):
    """
    Encoder------------------------------------------------------------

    기사 하나: (c1, c2, c3, c4)

    →   c1 embedding
        c2 embedding
        c3 embedding
        c4 embedding
        SEP

    history 전체:
        c1 c2 c3 c4 SEP
        c1 c2 c3 c4 SEP
        c1 c2 c3 c4 SEP
        ...
    
    Decoder training------------------------------------------------------------

    target SID: (2, 6, 7, 0) 라고 치면

    teacher forcing decoder input: [BOS, c1=2, c2=6, c3=7]

    각 position의 prediction:
        BOS           -> c1
        BOS c1        -> c2
        BOS c1 c2     -> c3
        BOS c1 c2 c3  -> c4

    즉 Decoder output position:
        position 0 -> c1_head
        position 1 -> c2_head
        position 2 -> c3_head
        position 3 -> c4_head
    """

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

        if c1_vocab_size <= 0:
            raise ValueError("c1_vocab_size must be > 0.")

        if c2_vocab_size <= 0:
            raise ValueError("c2_vocab_size must be > 0.")

        if c3_vocab_size <= 0:
            raise ValueError("c3_vocab_size must be > 0.")

        if c4_vocab_size <= 0:
            raise ValueError("c4_vocab_size must be > 0.")

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

        self.num_sid_levels = 4

        # ----------------------------------------------------
        # SID Embeddings
        #
        # c1/c2/c3/c4는 서로 의미도 다르고 vocabulary size도 다르므로 embedding table을 각각 따로 둔다.
        #
        # 예:
        # c1의 index 6과
        # c2의 index 6은 완전히 다른 embedding vector.
        # ----------------------------------------------------

        self.c1_embedding = nn.Embedding(
            num_embeddings=c1_vocab_size,
            embedding_dim=d_model,
        )

        self.c2_embedding = nn.Embedding(
            num_embeddings=c2_vocab_size,
            embedding_dim=d_model,
        )

        self.c3_embedding = nn.Embedding(
            num_embeddings=c3_vocab_size,
            embedding_dim=d_model,
        )

        self.c4_embedding = nn.Embedding(
            num_embeddings=c4_vocab_size,
            embedding_dim=d_model,
        )

        # BOS token : Decoder가 첫 c1을 예측하기 위한 시작 embedding.

        self.bos_embedding = nn.Parameter(
            torch.empty(1, 1, d_model)
        )

        # ----------------------------------------------------
        # SEP token
        # Encoder에서 기사와 기사를 구분.
        #
        # 기사1: c1 c2 c3 c4 SEP
        # 기사2: c1 c2 c3 c4 SEP
        # ----------------------------------------------------

        if self.use_sep:
            self.sep_embedding = nn.Parameter(
                torch.empty(1, 1, d_model)
            )
        else:
            self.register_parameter(
                "sep_embedding",
                None,
            )

        # T5 Encoder config
        #

        encoder_config = T5Config(
            vocab_size=1,
            d_model=d_model, #Transformer 내부 hidden vector의 차원 (=c1/c2/c3/c4 embedding의 차원)
            d_kv=d_model // num_heads, #Multi-Head Attention에서 각 어텐션 헤드가 사용하는 k,v 값 (d_model=256, num_heads=4면 각 헤드는 64차원)
            d_ff=d_ff, #Transformer block 내부 Feed Forward Network의 hidden 차원( attention을 거친 뒤 적용되는 MLP의 중간 차원)
            num_layers=num_layers, #Encoder Transformer block을 몇 층 쌓을지
            num_decoder_layers=num_layers, # Decoder 층 개수 
            num_heads=num_heads, # Multi-Head Attention의 head 개수
            dropout_rate=dropout_rate,

            is_decoder=False, #이 T5Stack을 Decoder가 아니라 Encoder로 사용한다는 뜻
            is_encoder_decoder=False, #Encoder와 Decoder를 각각 따로 만들기 때문에 False
            use_cache=False,
        )

        # ----------------------------------------------------
        # T5 Decoder config
        # is_decoder=True 이므로
        # self-attention + encoder-decoder cross attention 사용.
        # ----------------------------------------------------

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
        )

        # ----------------------------------------------------
        # Dummy embedding
        #T5Stack이 생성될 때 embed_tokens를 요구해서 형식상 넣어두는 placeholder이고, 실제 SID 입력은 inputs_embeds로 직접 넣기 때문에 사용되
        # ----------------------------------------------------

        self.encoder_dummy_embedding = nn.Embedding(
            1,
            d_model,
        )

        self.decoder_dummy_embedding = nn.Embedding(
            1,
            d_model,
        )

        # ----------------------------------------------------
        # Encoder / Decoder
        # ----------------------------------------------------

        self.encoder = T5Stack(
            encoder_config,
            embed_tokens=self.encoder_dummy_embedding,
        )

        self.decoder = T5Stack(
            decoder_config,
            embed_tokens=self.decoder_dummy_embedding,
        )

        # ----------------------------------------------------
        # Output Heads
        #
        # Decoder hidden state -> 각 hierarchy vocabulary logits
        # ----------------------------------------------------

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

        self.c4_head = nn.Linear(
            d_model,
            c4_vocab_size,
        )

        self._reset_parameters()


    def _reset_parameters(self) -> None:
        """
        SID / BOS / SEP embedding 초기화
        T5Stack 내부 weight는 HuggingFace에서 자체 초기화
        """

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

    # Input validation
    def _validate_history(
        self,
        history_sids: Tensor,
        history_mask: Tensor,
    ) -> None:

        if history_sids.ndim != 3:
            raise ValueError(
                "history_sids must have shape [B, H, 4]. "
                f"Received: {tuple(history_sids.shape)}"
            )

        if history_sids.shape[-1] != 4:
            raise ValueError(
                "Last dimension of history_sids must be 4 "
                "(c1, c2, c3, c4). "
                f"Received: {history_sids.shape[-1]}"
            )

        if history_mask.ndim != 2:
            raise ValueError(
                "history_mask must have shape [B, H]. "
                f"Received: {tuple(history_mask.shape)}"
            )

        if history_sids.shape[:2] != history_mask.shape:
            raise ValueError(
                "history_sids [B,H] and history_mask [B,H] "
                "dimensions must match."
            )

    def _validate_target(
        self,
        target_sids: Tensor,
    ) -> None:

        if target_sids.ndim != 2:
            raise ValueError(
                "target_sids must have shape [B, 4]. "
                f"Received: {tuple(target_sids.shape)}"
            )

        if target_sids.shape[-1] != 4:
            raise ValueError(
                "target_sids must contain exactly "
                "c1, c2, c3, c4."
            )

    # Embed history

    def embed_history(
        self,
        history_sids: Tensor,
    ) -> Tensor:
        """
        history_sids:
            [B, H, 4]

        Returns
        -------

        use_sep=True:
            [B, H*5, d_model]

            각 기사:
                c1 c2 c3 c4 SEP

        use_sep=False:
            [B, H*4, d_model]
        """

        # 각 hierarchy별 SID
        c1 = history_sids[:, :, 0]
        c2 = history_sids[:, :, 1]
        c3 = history_sids[:, :, 2]
        c4 = history_sids[:, :, 3]

        c1_emb = self.c1_embedding(c1)
        c2_emb = self.c2_embedding(c2)
        c3_emb = self.c3_embedding(c3)
        c4_emb = self.c4_embedding(c4)

        article_embeddings = torch.stack(
            [
                c1_emb,
                c2_emb,
                c3_emb,
                c4_emb,
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
            )

            sep = sep.unsqueeze(2)

            article_embeddings = torch.cat(
                [
                    article_embeddings,
                    sep,
                ],
                dim=2,
            )

        batch_size = article_embeddings.shape[0]

        encoder_inputs = article_embeddings.reshape(
            batch_size,
            -1,
            self.d_model,
        )

        return encoder_inputs

    # Expand history mask
    def expand_history_mask(
        self,
        history_mask: Tensor,
    ) -> Tensor:
        """
        sequence.py의 article-level mask를
        encoder token-level mask로 확장.

        Input: history_mask [B,H]
        예: [True, True, False]

        use_sep=True 이면:
            기사당 token = 5
            [
                T,T,T,T,T,
                T,T,T,T,T,
                F,F,F,F,F
            ]
        Returns:
            [B, encoder_seq_len]
        """

        tokens_per_article = (
            5
            if self.use_sep
            else 4
        )

        expanded_mask = history_mask.unsqueeze(-1).expand(
            -1,
            -1,
            tokens_per_article,
        )

        expanded_mask = expanded_mask.reshape(
            history_mask.shape[0],
            -1,
        )

        return expanded_mask

    # Encoder : user history를 인코더에 통과
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
            )
        )

        encoder_attention_mask = (
            encoder_attention_mask.long()
        )

        # T5 Encoder
        encoder_outputs = self.encoder(
            inputs_embeds=encoder_inputs,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        return EncoderOutput(
            hidden_states=encoder_outputs.last_hidden_state,
            attention_mask=encoder_attention_mask,
        )

    # Teacher-forcing decoder input
    def build_teacher_forcing_inputs(
        self,
        target_sids: Tensor,
    ) -> Tensor:
        """
        target: [c1, c2, c3, c4]

        Decoder input: [BOS, c1, c2, c3]

        c4는 입력에 X
        c4는 마지막에 예측해야 하기 때문.
        """

        self._validate_target(
            target_sids
        )

        batch_size = target_sids.shape[0]

        # BOS [B,1,D]
        bos = self.bos_embedding.expand(
            batch_size,
            -1,
            -1,
        )

        # target c1/c2/c3 embeddings : 각각 [B,D]
        c1_emb = self.c1_embedding(
            target_sids[:, 0]
        )

        c2_emb = self.c2_embedding(
            target_sids[:, 1]
        )

        c3_emb = self.c3_embedding(
            target_sids[:, 2]
        )

        # [B,1,D]
        c1_emb = c1_emb.unsqueeze(1)
        c2_emb = c2_emb.unsqueeze(1)
        c3_emb = c3_emb.unsqueeze(1)

        # ----------------------------------------------------
        # [B,4,D]
        #
        # position 0 = BOS
        # position 1 = c1
        # position 2 = c2
        # position 3 = c3
        # ----------------------------------------------------

        decoder_inputs = torch.cat(
            [
                bos,
                c1_emb,
                c2_emb,
                c3_emb,
            ],
            dim=1,
        )

        return decoder_inputs

    # Decoder
    def decode(
        self,
        decoder_inputs: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        """
        T5 Decoder는 is_decoder=True이므로
        causal self-attention이 자동으로 적용됨
        """

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs,

            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,

            use_cache=False,
            return_dict=True,
        )

        return decoder_outputs.last_hidden_state

    # Forward = TRAINING
    def forward(
        self,
        history_sids: Tensor,
        history_mask: Tensor,
        target_sids: Tensor,
    ) -> TransformerModelOutput:
        """
        sequence.py batch에서 그대로:
            history_sids = batch["history_sids"]
            history_mask = batch["history_mask"]
            target_sids  = batch["target_sids"]
        """

        self._validate_target(
            target_sids
        )
        # 1. Encode history
        encoder_output = self.encode(
            history_sids=history_sids,
            history_mask=history_mask,
        )
        # 2. Teacher-forcing decoder input
        # target = [c1,c2,c3,c4]
        # decoder input = [BOS,c1,c2,c3]

        decoder_inputs = (
            self.build_teacher_forcing_inputs(
                target_sids
            )
        )
        # 3. Decoder
        # output: [B,4,D]
        decoder_hidden = self.decode(
            decoder_inputs=decoder_inputs,
            encoder_hidden_states=encoder_output.hidden_states,
            encoder_attention_mask=encoder_output.attention_mask,
        )

        # 4. 각 position이 담당하는 hierarchy 예측
        c1_hidden = decoder_hidden[:, 0, :]
        c2_hidden = decoder_hidden[:, 1, :]
        c3_hidden = decoder_hidden[:, 2, :]
        c4_hidden = decoder_hidden[:, 3, :]

        c1_logits = self.c1_head(
            c1_hidden
        )

        c2_logits = self.c2_head(
            c2_hidden
        )

        c3_logits = self.c3_head(
            c3_hidden
        )

        c4_logits = self.c4_head(
            c4_hidden
        )

        return TransformerModelOutput(
            c1_logits=c1_logits,
            c2_logits=c2_logits,
            c3_logits=c3_logits,
            c4_logits=c4_logits,
        )

    # Prefix embedding for inference
    def build_prefix_inputs(
        self,
        prefix_sids: Optional[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """
        predict_sid.py에서 autoregressive generation할 때 사용

        prefix 길이에 따라:

        prefix 없음:
            [BOS]
            -> c1 예측

        prefix=[c1]:
            [BOS,c1]
            -> c2 예측

        prefix=[c1,c2]:
            [BOS,c1,c2]
            -> c3 예측

        prefix=[c1,c2,c3]:
            [BOS,c1,c2,c3]
            -> c4 예측


        prefix_sids shape:
            [B,L]

        L은 0~3
        """

        bos = self.bos_embedding.expand(
            batch_size,
            -1,
            -1,
        )

        if prefix_sids is None:
            return bos

        if prefix_sids.ndim != 2:
            raise ValueError(
                "prefix_sids must have shape [B,L]."
            )

        if prefix_sids.shape[0] != batch_size:
            raise ValueError(
                "prefix_sids batch size does not match "
                "encoder batch size."
            )

        prefix_length = prefix_sids.shape[1]

        if prefix_length > 3:
            raise ValueError(
                "Prefix can contain at most c1,c2,c3. "
                "After c4 the SID is already complete."
            )

        if prefix_length == 0:
            return bos

        prefix_embeddings = []

        # prefix position 0 = c1
        if prefix_length >= 1:
            prefix_embeddings.append(
                self.c1_embedding(
                    prefix_sids[:, 0]
                ).unsqueeze(1)
            )

        # prefix position 1 = c2
        if prefix_length >= 2:
            prefix_embeddings.append(
                self.c2_embedding(
                    prefix_sids[:, 1]
                ).unsqueeze(1)
            )

        # prefix position 2 = c3
        if prefix_length >= 3:
            prefix_embeddings.append(
                self.c3_embedding(
                    prefix_sids[:, 2]
                ).unsqueeze(1)
            )

        return torch.cat(
            [bos] + prefix_embeddings,
            dim=1,
        )

    # Next-token prediction for inference
    def next_token_logits(
        self,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
        prefix_sids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        predict_sid.py에서 다음 SID level 하나를 예측할 때 사용
        prefix=None
            -> c1 logits

        prefix=[c1]
            -> c2 logits

        prefix=[c1,c2]
            -> c3 logits

        prefix=[c1,c2,c3]
            -> c4 logits

        Returns -------
        다음 hierarchy의 logits:

            c1: [B,c1_vocab_size]
            c2: [B,c2_vocab_size]
            c3: [B,c3_vocab_size]
            c4: [B,c4_vocab_size]
        """

        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device

        if prefix_sids is None:
            prefix_length = 0

        else:
            prefix_sids = prefix_sids.to(
                device=device,
                dtype=torch.long,
            )

            prefix_length = prefix_sids.shape[1]

        # [BOS + prefix]
        decoder_inputs = self.build_prefix_inputs(
            prefix_sids=prefix_sids,
            batch_size=batch_size,
            device=device,
        )

        # Decoder
        decoder_hidden = self.decode(
            decoder_inputs=decoder_inputs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        # 가장 마지막 position에서 다음 code를 예측
        last_hidden = decoder_hidden[:, -1, :]

        # prefix 길이에 따라 다음 hierarchy 결정
        if prefix_length == 0:

            # BOS -> c1
            return self.c1_head(
                last_hidden
            )

        elif prefix_length == 1:

            # BOS,c1 -> c2
            return self.c2_head(
                last_hidden
            )

        elif prefix_length == 2:

            # BOS,c1,c2 -> c3
            return self.c3_head(
                last_hidden
            )

        elif prefix_length == 3:

            # BOS,c1,c2,c3 -> c4
            return self.c4_head(
                last_hidden
            )

        else:
            raise ValueError(
                f"Invalid prefix length: {prefix_length}"
            )


if __name__ == "__main__":
    #테스트 샘플
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
                [0, 0, 0, 0],  # padding
            ],
        ],
        dtype=torch.long,
    )

    history_mask = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
        ],
        dtype=torch.bool,
    )

    target_sids = torch.tensor(
        [
            [2, 6, 7, 0],
            [3, 4, 9, 1],
        ],
        dtype=torch.long,
    )

    model = NewsEncoderDecoderTransformer(
        c1_vocab_size=18,
        c2_vocab_size=128,
        c3_vocab_size=512,
        c4_vocab_size=32,

        d_model=256,
        num_heads=4,
        d_ff=1024,
        num_layers=4,
        dropout_rate=0.1,
        use_sep=True,
    )

    # Training forward
    output = model(
        history_sids=history_sids,
        history_mask=history_mask,
        target_sids=target_sids,
    )

    print("===== Training Forward =====")

    print(
        "c1_logits:",
        output.c1_logits.shape,
    )

    print(
        "c2_logits:",
        output.c2_logits.shape,
    )

    print(
        "c3_logits:",
        output.c3_logits.shape,
    )

    print(
        "c4_logits:",
        output.c4_logits.shape,
    )

    encoder_output = model.encode(
        history_sids=history_sids,
        history_mask=history_mask,
    )

    # BOS -> c1
    c1_logits = model.next_token_logits(
        encoder_hidden_states=encoder_output.hidden_states,
        encoder_attention_mask=encoder_output.attention_mask,
        prefix_sids=None,
    )

    pred_c1 = torch.argmax(
        c1_logits,
        dim=-1,
    )

    print()
    print("===== Inference =====")

    print(
        "predicted c1:",
        pred_c1,
    )

    prefix = pred_c1.unsqueeze(1)

    c2_logits = model.next_token_logits(
        encoder_hidden_states=encoder_output.hidden_states,
        encoder_attention_mask=encoder_output.attention_mask,
        prefix_sids=prefix,
    )

    pred_c2 = torch.argmax(
        c2_logits,
        dim=-1,
    )

    print(
        "predicted c2:",
        pred_c2,
    )

    prefix = torch.stack(
        [
            pred_c1,
            pred_c2,
        ],
        dim=1,
    )

    c3_logits = model.next_token_logits(
        encoder_hidden_states=encoder_output.hidden_states,
        encoder_attention_mask=encoder_output.attention_mask,
        prefix_sids=prefix,
    )

    pred_c3 = torch.argmax(
        c3_logits,
        dim=-1,
    )

    print(
        "predicted c3:",
        pred_c3,
    )

    prefix = torch.stack(
        [
            pred_c1,
            pred_c2,
            pred_c3,
        ],
        dim=1,
    )

    c4_logits = model.next_token_logits(
        encoder_hidden_states=encoder_output.hidden_states,
        encoder_attention_mask=encoder_output.attention_mask,
        prefix_sids=prefix,
    )

    pred_c4 = torch.argmax(
        c4_logits,
        dim=-1,
    )

    print(
        "predicted c4:",
        pred_c4,
    )

    predicted_sid = torch.stack(
        [
            pred_c1,
            pred_c2,
            pred_c3,
            pred_c4,
        ],
        dim=1,
    )

    print()
    print(
        "predicted SID:",
        predicted_sid,
    )