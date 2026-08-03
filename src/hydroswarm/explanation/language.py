"""Optional small decoder isolated from safety-critical operational heads."""

from __future__ import annotations

from torch import Tensor, nn


class ConstrainedLanguageDecoder(nn.Module):
    def __init__(
        self, *, vocab_size: int = 4096, d_model: int = 256, nhead: int = 8,
        dim_feedforward: int = 768, num_layers: int = 3, max_tokens: int = 128,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.operational_projection = nn.LazyLinear(d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, token_ids: Tensor, verified_operational_latent: Tensor) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] > self.max_tokens:
            raise ValueError("token_ids must be [batch, tokens] within max_tokens")
        language_memory = self.operational_projection(verified_operational_latent.detach())
        if language_memory.ndim == 2:
            language_memory = language_memory[:, None, :]
        target = self.embedding(token_ids)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            token_ids.shape[1], device=token_ids.device
        )
        return self.output(self.decoder(target, language_memory, tgt_mask=causal_mask))

