import torch
import sacrebleu

from .vocabulary import Vocabulary


@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, sos_idx, eos_idx, pad_idx, device):
    """Auto-regressive greedy decoding. Returns generated token IDs (B, T)."""
    model.eval()
    encoder_out = model.encoder(src, src_mask)

    batch_size = src.size(0)
    tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        T = tgt.size(1)
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(T, T, device=device)).bool()
        tgt_mask = pad_mask & causal_mask

        decoder_out = model.decoder(tgt, encoder_out, tgt_mask, src_mask)
        next_token = model.linear(decoder_out[:, -1]).argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)

        finished = finished | (next_token.squeeze(-1) == eos_idx)
        if finished.all():
            break

    return tgt


def compute_bleu(model, loader, tgt_vocab, max_len, device):
    """Compute corpus-level BLEU on a loader using greedy decoding."""
    sos_idx = tgt_vocab.word2idx[Vocabulary.SOS]
    eos_idx = tgt_vocab.word2idx[Vocabulary.EOS]
    pad_idx = tgt_vocab.word2idx[Vocabulary.PAD]

    model.eval()
    hypotheses, references = [], []

    for src, tgt in loader:
        src = src.to(device)
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

        output = greedy_decode(model, src, src_mask, max_len, sos_idx, eos_idx, pad_idx, device)

        for hyp_ids, ref_ids in zip(output.tolist(), tgt.tolist()):
            hypotheses.append(tgt_vocab.decode(hyp_ids))
            references.append(tgt_vocab.decode(ref_ids))

    bleu = sacrebleu.corpus_bleu(hypotheses, [references], force=True)
    return bleu.score
