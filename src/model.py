import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model, N, heads, d_ff, drop_rate, max_len):
        super().__init__()
        self.encoder = Encoder(source_vocab_size, d_model, N, heads, d_ff, drop_rate, max_len)
        self.decoder = Decoder(target_vocab_size, d_model, N, heads, d_ff, drop_rate, max_len)

        self.linear = nn.Linear(d_model, target_vocab_size, bias=False)

        self._init_weights()

        # Tie target embedding and output projection (Press & Wolf, 2017)
        self.linear.weight = self.decoder.embedding.weight

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.encoder.embedding.weight, mean=0.0, std=self.encoder.d_model ** -0.5)
        nn.init.normal_(self.decoder.embedding.weight, mean=0.0, std=self.decoder.d_model ** -0.5)

    def forward(self, source, target, source_mask, target_mask):
        """
        Args:
            source:      Source sequence (e.g., English IDs)
            target:      Target sequence (e.g., German IDs - shifted right)
            source_mask: Padding mask for the encoder and decoder cross-attention
            target_mask: Look-ahead mask for the decoder self-attention
        """
        encoder_out = self.encoder(source, source_mask)
        decoder_out = self.decoder(target, encoder_out, target_mask, source_mask)
        logits = self.linear(decoder_out)
        return logits
