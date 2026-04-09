"""Load AVASR components and reconstruct them from saved state dicts."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def _load_model_config(weights_dir: Path) -> dict:
    with open(weights_dir / "model_config.json") as f:
        return json.load(f)


def load_preprocessor(weights_dir: Path, device: str = "cpu"):
    from src.avasr.core.mel_features import AudioToMelSpectrogramPreprocessor
    pcfg = _load_model_config(weights_dir)["preprocessor"]
    preprocessor = AudioToMelSpectrogramPreprocessor(
        sample_rate=pcfg.get("sample_rate", 16000),
        window_size=pcfg.get("window_size", 0.025),
        window_stride=pcfg.get("window_stride", 0.01),
        window=pcfg.get("window", "hann"),
        features=pcfg.get("features", 128),
        n_fft=pcfg.get("n_fft", 512),
        dither=pcfg.get("dither", 1e-5),
        pad_to=pcfg.get("pad_to", 0),
        normalize=pcfg.get("normalize", "per_feature"),
    )
    preprocessor.load_state_dict(torch.load(weights_dir / "preprocessor.pt", weights_only=True, map_location="cpu"))
    preprocessor.to(device).eval()
    return preprocessor


def load_encoder(weights_dir: Path, device: str = "cpu"):
    from src.avasr.core.conformer import ConformerEncoderSTNOAV
    cfg = _load_model_config(weights_dir)
    ecfg = cfg["encoder"]
    encoder = ConformerEncoderSTNOAV(
        feat_in=cfg["preprocessor"].get("features", 128),
        feat_out=-1,
        n_layers=ecfg.get("n_layers", 24),
        d_model=ecfg.get("d_model", 1024),
        use_bias=ecfg.get("use_bias", False),
        subsampling=ecfg.get("subsampling", "dw_striding"),
        subsampling_factor=ecfg.get("subsampling_factor", 8),
        subsampling_conv_channels=ecfg.get("subsampling_conv_channels", 256),
        ff_expansion_factor=ecfg.get("ff_expansion_factor", 4),
        self_attention_model=ecfg.get("self_attention_model", "rel_pos"),
        n_heads=ecfg.get("n_heads", 8),
        att_context_size=list(ecfg.get("att_context_size", [-1, -1])),
        xscaling=ecfg.get("xscaling", False),
        untie_biases=ecfg.get("untie_biases", True),
        pos_emb_max_len=ecfg.get("pos_emb_max_len", 45000),
        conv_kernel_size=ecfg.get("conv_kernel_size", 9),
        conv_norm_type=ecfg.get("conv_norm_type", "batch_norm"),
        dropout=ecfg.get("dropout", 0.1),
        dropout_pre_encoder=ecfg.get("dropout_pre_encoder", 0.1),
        dropout_emb=ecfg.get("dropout_emb", 0.0),
        dropout_att=ecfg.get("dropout_att", 0.1),
        use_pre_pe_visual_conditioning=ecfg.get("use_pre_pe_visual_conditioning", True),
        use_visual_conditioning_on_all_layers=ecfg.get("use_visual_conditioning_on_all_layers", True),
        visual_conditioning_method=ecfg.get("visual_conditioning_method", "non_lin_cross_fade"),
        visual_preprocessing_model=ecfg.get("visual_preprocessing_model", "base"),
        conditioning_embed_aggr_method=ecfg.get("conditioning_embed_aggr_method", "softmax_wavg"),
        num_conditioning_embeds=ecfg.get("num_conditioning_embeds", 25),
        d_visual_embeds=ecfg.get("d_visual_embeds", 1024),
        modality_dropout_prob=ecfg.get("modality_dropout_prob", 0.0),
        use_visual_adapter_encoder=ecfg.get("use_visual_adapter_encoder", False),
        share_visual_preprocessing=ecfg.get("share_visual_preprocessing", False),
        use_stno=ecfg.get("use_stno", False),
        stochastic_depth_drop_prob=ecfg.get("stochastic_depth_drop_prob", 0.0),
        stochastic_depth_mode=ecfg.get("stochastic_depth_mode", "linear"),
        stochastic_depth_start_layer=ecfg.get("stochastic_depth_start_layer", 1),
    )
    encoder.load_state_dict(torch.load(weights_dir / "encoder.pt", weights_only=True, map_location="cpu"))
    encoder.to(device).eval()
    return encoder


def load_decoder(weights_dir: Path, device: str = "cpu"):
    from src.avasr.core.rnnt import RNNTDecoder
    cfg = _load_model_config(weights_dir)
    with open(weights_dir / "tokenizer" / "tokenizer_info.json") as f:
        vocab_size = json.load(f)["vocab_size"]
    decoder = RNNTDecoder(
        prednet={"pred_hidden": cfg["model_defaults"].get("pred_hidden", 640), "pred_rnn_layers": 2, "dropout": 0.2},
        vocab_size=vocab_size,
        blank_as_pad=True,
        normalization_mode=None,
    )
    decoder.load_state_dict(torch.load(weights_dir / "decoder.pt", weights_only=True, map_location="cpu"))
    decoder.to(device).eval()
    return decoder


def load_joint(weights_dir: Path, device: str = "cpu"):
    from src.avasr.core.rnnt import RNNTJoint
    cfg = _load_model_config(weights_dir)
    with open(weights_dir / "tokenizer" / "tokenizer_info.json") as f:
        vocab_size = json.load(f)["vocab_size"]
    joint = RNNTJoint(
        jointnet={
            "joint_hidden": cfg["model_defaults"].get("joint_hidden", 640),
            "activation": "relu",
            "dropout": 0.2,
            "encoder_hidden": cfg["encoder"].get("d_model", 1024),
            "pred_hidden": cfg["model_defaults"].get("pred_hidden", 640),
        },
        num_classes=vocab_size,
        num_extra_outputs=cfg["model_defaults"].get("num_tdt_durations", 5),
        log_softmax=None,
    )
    joint.load_state_dict(torch.load(weights_dir / "joint.pt", weights_only=True, map_location="cpu"))
    joint.to(device).eval()
    return joint


def load_tokenizer(weights_dir: Path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(weights_dir / "tokenizer" / "tokenizer.model"))
    return sp

