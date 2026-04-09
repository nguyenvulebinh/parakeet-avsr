"""HuggingFace-style PreTrainedModel for AV-ASR (Audio-Visual ASR)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import PreTrainedModel

from src.avasr.core.configuration import AVASRConfig
from src.avasr.core.conformer import ConformerEncoderSTNOAV
from src.avasr.core.mel_features import AudioToMelSpectrogramPreprocessor
from src.avasr.core.rnnt import RNNTDecoder, RNNTJoint
from src.avasr.core.tdt_decode import TDTResult, greedy_tdt_decode_batch
from src.avasr.vision.avhubert import AVHubertAVSR, AVHubertAVSRConfig


class AVASRForTranscription(PreTrainedModel):
    config_class = AVASRConfig
    _supports_param_buffer_assignment = False

    def __init__(self, config: AVASRConfig):
        super().__init__(config)
        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=config.sample_rate,
            window_size=config.window_size,
            window_stride=config.window_stride,
            window=config.window,
            features=config.features,
            n_fft=config.n_fft,
            dither=config.dither,
            pad_to=config.pad_to,
            normalize=config.normalize,
        )
        self.encoder = ConformerEncoderSTNOAV(
            feat_in=config.features,
            feat_out=-1,
            n_layers=config.n_layers,
            d_model=config.d_model,
            use_bias=config.use_bias,
            subsampling=config.subsampling,
            subsampling_factor=config.subsampling_factor,
            subsampling_conv_channels=config.subsampling_conv_channels,
            ff_expansion_factor=config.ff_expansion_factor,
            self_attention_model=config.self_attention_model,
            n_heads=config.n_heads,
            att_context_size=list(config.att_context_size),
            xscaling=config.xscaling,
            untie_biases=config.untie_biases,
            pos_emb_max_len=config.pos_emb_max_len,
            conv_kernel_size=config.conv_kernel_size,
            conv_norm_type=config.conv_norm_type,
            dropout=config.dropout,
            dropout_pre_encoder=config.dropout_pre_encoder,
            dropout_emb=config.dropout_emb,
            dropout_att=config.dropout_att,
            stochastic_depth_drop_prob=config.stochastic_depth_drop_prob,
            stochastic_depth_mode=config.stochastic_depth_mode,
            stochastic_depth_start_layer=config.stochastic_depth_start_layer,
            d_visual_embeds=config.d_visual_embeds,
            visual_conditioning_method=config.visual_conditioning_method,
            use_pre_pe_visual_conditioning=config.use_pre_pe_visual_conditioning,
            use_visual_conditioning_on_all_layers=config.use_visual_conditioning_on_all_layers,
            visual_preprocessing_model=config.visual_preprocessing_model,
            conditioning_embed_aggr_method=config.conditioning_embed_aggr_method,
            num_conditioning_embeds=config.num_conditioning_embeds,
            modality_dropout_prob=config.modality_dropout_prob,
            use_visual_adapter_encoder=config.use_visual_adapter_encoder,
            share_visual_preprocessing=config.share_visual_preprocessing,
            use_stno=config.use_stno,
        )
        self.decoder = RNNTDecoder(
            prednet={
                "pred_hidden": config.pred_hidden,
                "pred_rnn_layers": config.pred_rnn_layers,
                "dropout": config.pred_dropout,
            },
            vocab_size=config.vocab_size,
            blank_as_pad=True,
            normalization_mode=None,
        )
        self.joint = RNNTJoint(
            jointnet={
                "joint_hidden": config.joint_hidden,
                "activation": config.joint_activation,
                "dropout": config.joint_dropout,
                "encoder_hidden": config.d_model,
                "pred_hidden": config.pred_hidden,
            },
            num_classes=config.vocab_size,
            num_extra_outputs=config.num_tdt_durations,
            log_softmax=None,
        )
        self.vis_feat_extractor = (
            AVHubertAVSR(AVHubertAVSRConfig(**config.avhubert_config))
            if config.avhubert_config
            else None
        )
        self.post_init()
        self._reinit_buffers()

    def _reinit_buffers(self):
        for name, buf in self.named_buffers():
            if buf.device.type == "meta":
                module_path, _, _ = name.rpartition(".")
                mod = self
                for part in module_path.split("."):
                    mod = getattr(mod, part)
                if hasattr(mod, "extend_pe"):
                    mod.extend_pe(
                        buf.size(1),
                        device=torch.device("cpu"),
                        dtype=buf.dtype if buf.dtype != torch.float32 else torch.float32,
                    )

    def forward(
        self,
        audio_signal: torch.Tensor,
        audio_lengths: torch.Tensor,
        visual_embeds: Optional[torch.Tensor] = None,
        visual_embed_lengths: Optional[torch.Tensor] = None,
        num_speakers: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mel, mel_len = self.preprocessor(input_signal=audio_signal, length=audio_lengths)
        encoded, encoded_len = self.encoder(
            audio_signal=mel,
            length=mel_len,
            visual_embeds=visual_embeds,
            visual_embed_lengths=visual_embed_lengths,
            num_speakers=num_speakers,
        )
        return encoded, encoded_len

    @torch.no_grad()
    def decode(self, encoded: torch.Tensor, encoded_len: torch.Tensor) -> list[TDTResult]:
        return greedy_tdt_decode_batch(
            encoded,
            encoded_len,
            self.decoder,
            self.joint,
            blank_id=self.config.blank_id,
            durations=self.config.tdt_durations,
            max_symbols=self.config.max_symbols,
        )

    @torch.no_grad()
    def transcribe(self, av_path: str | Path) -> str:
        from src.avasr.io.audio_loader import load_audio
        from src.avasr.io.video_loader import load_video_frames
        from src.avasr.vision.visual_features import get_visual_feats

        device = self.device
        audio = load_audio(av_path).to(device)
        _, lip_frames, _ = load_video_frames(av_path)
        video_batch = lip_frames.unsqueeze(0).to(device)
        video_lengths = torch.tensor([lip_frames.shape[0]], dtype=torch.int64, device=device)
        num_speakers = torch.tensor([1], dtype=torch.int64, device=device)
        visual_embeds = get_visual_feats(
            self.vis_feat_extractor,
            video_batch,
            video_lengths,
            num_speakers=num_speakers,
            extract_all_layers=self.config.extract_visual_features_all_layers,
            chunk_length=self.config.visual_chunk_length_sec,
        )
        audio_batch = audio.unsqueeze(0)
        audio_lengths = torch.tensor([audio.shape[0]], dtype=torch.int64, device=device)
        encoded, encoded_len = self.forward(
            audio_signal=audio_batch,
            audio_lengths=audio_lengths,
            visual_embeds=visual_embeds,
            visual_embed_lengths=video_lengths,
            num_speakers=num_speakers,
        )
        results = self.decode(encoded, encoded_len)
        return self._get_tokenizer().decode(results[0].tokens)

    def _get_tokenizer(self):
        if not hasattr(self, "_sp_tokenizer") or self._sp_tokenizer is None:
            import sentencepiece as spm
            from transformers.utils.hub import cached_file

            model_dir = Path(self.config.name_or_path)
            if model_dir.exists():
                tok_path = model_dir / "tokenizer.model"
            else:
                tok_path = cached_file(
                    self.config.name_or_path,
                    "tokenizer.model",
                )
                if tok_path is None:
                    raise FileNotFoundError(
                        f"tokenizer.model not found for {self.config.name_or_path}"
                    )
            self._sp_tokenizer = spm.SentencePieceProcessor()
            self._sp_tokenizer.load(str(tok_path))
        return self._sp_tokenizer

