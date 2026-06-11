import torch

from melocodec import BWC


def main():
    model = BWC(
        encoder_dim=8,
        encoder_rates=(2, 2),
        decoder_dim=32,
        decoder_rates=(2, 2),
        n_codebooks=2,
        codebook_size=32,
        codebook_dim=4,
        sample_rate=16000,
    )
    wav = torch.randn(1, 1, 4096)
    out = model(wav, n_quantizers=1)
    print("audio:", tuple(out["audio"].shape))
    print("codes:", tuple(out["codes"].shape))
    print("chroma loss:", float(out["chroma/recon_loss"]))


if __name__ == "__main__":
    main()
