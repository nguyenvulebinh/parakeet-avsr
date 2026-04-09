"""HuggingFace-style configuration for AV-ASR (Audio-Visual Automatic Speech Recognition)."""
from __future__ import annotations

from transformers import PretrainedConfig


class AVASRConfig(PretrainedConfig):
    model_type = "avasr"

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 0.025,
        window_stride: float = 0.01,
        window: str = "hann",
        features: int = 128,
        n_fft: int = 512,
        dither: float = 1e-5,
        pad_to: int = 0,
        normalize: str = "per_feature",
        n_layers: int = 24,
        d_model: int = 1024,
        use_bias: bool = False,
        subsampling: str = "dw_striding",
        subsampling_factor: int = 8,
        subsampling_conv_channels: int = 256,
        ff_expansion_factor: int = 4,
        self_attention_model: str = "rel_pos",
        n_heads: int = 8,
        att_context_size: list[int] | None = None,
        xscaling: bool = False,
        untie_biases: bool = True,
        pos_emb_max_len: int = 45000,
        conv_kernel_size: int = 9,
        conv_norm_type: str = "batch_norm",
        dropout: float = 0.1,
        dropout_pre_encoder: float = 0.1,
        dropout_emb: float = 0.0,
        dropout_att: float = 0.1,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        d_visual_embeds: int = 1024,
        visual_conditioning_method: str = "non_lin_cross_fade",
        use_pre_pe_visual_conditioning: bool = True,
        use_visual_conditioning_on_all_layers: bool = True,
        visual_preprocessing_model: str = "base",
        conditioning_embed_aggr_method: str = "softmax_wavg",
        num_conditioning_embeds: int = 25,
        modality_dropout_prob: float = 0.0,
        use_visual_adapter_encoder: bool = False,
        share_visual_preprocessing: bool = False,
        use_stno: bool = False,
        avhubert_config: dict | None = None,
        extract_visual_features_all_layers: bool = True,
        visual_chunk_length_sec: int = 20,
        pred_hidden: int = 640,
        pred_rnn_layers: int = 2,
        pred_dropout: float = 0.2,
        joint_hidden: int = 640,
        joint_activation: str = "relu",
        joint_dropout: float = 0.2,
        num_tdt_durations: int = 5,
        tdt_durations: list[int] | None = None,
        max_symbols: int = 10,
        vocab_size: int = 1024,
        blank_id: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.features = features
        self.n_fft = n_fft
        self.dither = dither
        self.pad_to = pad_to
        self.normalize = normalize
        self.n_layers = n_layers
        self.d_model = d_model
        self.use_bias = use_bias
        self.subsampling = subsampling
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.ff_expansion_factor = ff_expansion_factor
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.att_context_size = att_context_size or [-1, -1]
        self.xscaling = xscaling
        self.untie_biases = untie_biases
        self.pos_emb_max_len = pos_emb_max_len
        self.conv_kernel_size = conv_kernel_size
        self.conv_norm_type = conv_norm_type
        self.dropout = dropout
        self.dropout_pre_encoder = dropout_pre_encoder
        self.dropout_emb = dropout_emb
        self.dropout_att = dropout_att
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob
        self.stochastic_depth_mode = stochastic_depth_mode
        self.stochastic_depth_start_layer = stochastic_depth_start_layer
        self.d_visual_embeds = d_visual_embeds
        self.visual_conditioning_method = visual_conditioning_method
        self.use_pre_pe_visual_conditioning = use_pre_pe_visual_conditioning
        self.use_visual_conditioning_on_all_layers = use_visual_conditioning_on_all_layers
        self.visual_preprocessing_model = visual_preprocessing_model
        self.conditioning_embed_aggr_method = conditioning_embed_aggr_method
        self.num_conditioning_embeds = num_conditioning_embeds
        self.modality_dropout_prob = modality_dropout_prob
        self.use_visual_adapter_encoder = use_visual_adapter_encoder
        self.share_visual_preprocessing = share_visual_preprocessing
        self.use_stno = use_stno
        self.avhubert_config = avhubert_config
        self.extract_visual_features_all_layers = extract_visual_features_all_layers
        self.visual_chunk_length_sec = visual_chunk_length_sec
        self.pred_hidden = pred_hidden
        self.pred_rnn_layers = pred_rnn_layers
        self.pred_dropout = pred_dropout
        self.joint_hidden = joint_hidden
        self.joint_activation = joint_activation
        self.joint_dropout = joint_dropout
        self.num_tdt_durations = num_tdt_durations
        self.tdt_durations = tdt_durations or [0, 1, 2, 3, 4]
        self.max_symbols = max_symbols
        self.vocab_size = vocab_size
        self.blank_id = blank_id

