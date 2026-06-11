import torch

from melocodec import BWC, ChromaCodec, MeloCodec


def test_bwc_forward_smoke():
    melody_codec = ChromaCodec(
        sample_rate=16000,
        chroma_hop_length=128,
        chroma_n_fft=512,
        encoder_dims=(8, 16),
        decoder_dims=(16, 8),
        encoder_strides=(1, 2),
        decoder_strides=(2, 1),
        codebook_size=16,
        codebook_dim=4,
    )
    model = BWC(
        encoder_dim=8,
        encoder_rates=(2, 2),
        decoder_dim=32,
        decoder_rates=(2, 2),
        n_codebooks=2,
        codebook_size=16,
        codebook_dim=4,
        sample_rate=16000,
        melody_codec=melody_codec,
    )
    wav = torch.randn(2, 1, 4096)
    out = model(wav, n_quantizers=1)

    assert MeloCodec is BWC
    assert out["audio"].shape == wav.shape
    assert out["codes"].shape[0] == wav.shape[0]
    assert out["codes"].shape[1] == 1
    assert out["chroma/recon_loss"].ndim == 0
