import json
import math
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
import torchvision
from datasets import concatenate_datasets, load_dataset
from python_speech_features import logfbank
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from src.avasr.io.audio_loader import AUDIO_SAMPLE_RATE, load_audio
from src.avasr.io.video_loader import load_video_frames

def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data

def load_augment_audio(source, start_time=0, end_time=None):
    """
    Load audio waveform from various input types.

    Args:
        source: str (file path), bytes (raw file content),
                np.ndarray, or torch.Tensor (pre-loaded waveform).
        start_time: start time in seconds (file-based loading only).
        end_time: end time in seconds (file-based loading only).

    Returns:
        torch.Tensor of shape T x 1 (mono, 16 kHz expected).
    """
    if isinstance(source, torch.Tensor):
        audio = source.float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(1)
        return audio

    if isinstance(source, np.ndarray):
        audio = torch.from_numpy(source).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(1)
        return audio

    if isinstance(source, (bytes, bytearray)):
        import io
        waveform, sample_rate = torchaudio.load(io.BytesIO(source))
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        return waveform.transpose(1, 0)

    if AudioDecoder is not None:
        audio_decoder = AudioDecoder(source)
        if end_time is None:
            end_time = audio_decoder.metadata.duration_seconds_from_header
        waveform = audio_decoder.get_samples_played_in_range(start_time, end_time).data
    else:
        if start_time == 0 and end_time is None:
            frame_offset = 0
            num_frames = -1
        else:
            frame_offset = int(start_time * 16000)
            num_frames = int((end_time - start_time) * 16000)
        waveform, sample_rate = torchaudio.load(source, frame_offset=frame_offset, num_frames=num_frames, normalize=True)
        assert sample_rate == 16000
    return waveform.transpose(1, 0)  # T x 1

def filter_fn(ex: Dict[str, Any]) -> bool:
    if ex.get("duration", None) is None:
        return True
    if ex.get("duration", 0) <= 0.2:
        return False
    if ex.get("duration", 0) > 40.0:
        return False
    return True

class FBanksAndStack(torch.nn.Module):
    def __init__(self, stack_order=4):
        super().__init__()
        self.stack_order = stack_order

    def stacker(self, feats):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % self.stack_order != 0:
            res = self.stack_order - len(feats) % self.stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, self.stack_order, feat_dim)).reshape(-1, self.stack_order*feat_dim)
        return feats

    def forward(self, x):
        # x: T x 1
        # return: T x F*stack_order
        audio_feats = logfbank(x.squeeze().numpy(), samplerate=16000).astype(np.float32) # [T, F]
        audio_feats = self.stacker(audio_feats) # [T/stack_order_audio, F*stack_order_audio]
        
        with torch.no_grad():
            audio_feats = F.layer_norm(torch.from_numpy(audio_feats), audio_feats.shape[1:])
        return audio_feats

class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_filename=None,
        snr_target=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20, 999999]
        if noise_filename is None:
            # self.noise = torch.randn(1, 16000)
            self.noise = None
        else:
            self.noise, sample_rate = torchaudio.load(noise_filename)
            assert sample_rate == 16000

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        if self.noise is None:
            return speech
        speech = speech.t()
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx : start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.t()

class AddMultiSpk(torch.nn.Module):
    def __init__(
        self,
        speech_dataset=None,
        snr_target=None,
        interferer_spk=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20]
        self.interferer_spk = [interferer_spk] if interferer_spk else [0, 0, 1, 2]
        self.speech_dataset = speech_dataset

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        if self.speech_dataset is None:
            return speech
        speech_length = speech.size(0) / 16000
        if speech_length < 2:
            return speech
        
        num_interferer = random.choice(self.interferer_spk)
        interferer_signal = None
        for _ in range(num_interferer):
            interferer = load_augment_audio(random.choice(self.speech_dataset)['video'])
            interferer_length = interferer.size(0) / 16000
            if 2 <= interferer_length <= 10:
                interferer = cut_or_pad(interferer, len(speech))
                if interferer_signal is None:
                    interferer_signal = interferer
                else:
                    snr_level = torch.tensor([random.choice([-5, 0, 5, 10, 15])])
                    interferer_signal = torchaudio.functional.add_noise(interferer_signal.t(), interferer.t(), snr_level).t()        
        
        if interferer_signal is None:
            return speech
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        speech = torchaudio.functional.add_noise(speech.t(), interferer_signal.t(), snr_level).t()
        
        return speech

class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned

class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)

class AudioTransform:
    def __init__(self, subset="val", speech_dataset=None, snr_target=None):
        # print("subset\n", subset, "speech_dataset\n", speech_dataset, "snr_target\n", snr_target)
        if subset == "train":
            self.audio_pipeline = torch.nn.Sequential(
                AdaptiveTimeMask(6400, 16000),
                AddMultiSpk(speech_dataset=speech_dataset),
                AddNoise(),
                # FBanksAndStack(),
            )
        elif subset == "val" or subset == "test":
            self.audio_pipeline = torch.nn.Sequential(
                AddNoise(snr_target=snr_target)
                if snr_target is not None
                else FunctionalModule(lambda x: x),
                # FBanksAndStack(),
            )

    def __call__(self, sample):
        # sample: T x 1
        # rtype: T x 1
        return self.audio_pipeline(sample)

class DistributedSortedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, 
                 shuffle=True, seed=0, drop_last=False, lengths=None):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.lengths = lengths

    def __iter__(self):
        if self.lengths is not None:
            indices = torch.argsort(torch.tensor(self.lengths), descending=True).tolist()
        else:
            indices = list(range(len(self.dataset)))

        mega_batch_size = self.batch_size * self.num_replicas
        batches = []

        for i in range(0, len(indices), mega_batch_size):
            mega_batch = indices[i : i + mega_batch_size]

            if self.drop_last and len(mega_batch) < mega_batch_size:
                break

            start_idx = self.rank * self.batch_size
            end_idx = start_idx + self.batch_size
            rank_batch = mega_batch[start_idx:end_idx]

            if len(rank_batch) > 0:
                batches.append(rank_batch)

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            shuffle_idx = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in shuffle_idx]

        return iter(batches)

    def __len__(self):
        mega_batch_size = self.batch_size * self.num_replicas
        if self.drop_last:
            return len(self.dataset) // mega_batch_size
        else:
            return math.ceil(len(self.dataset) / mega_batch_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

class AVASRDataset(Dataset):
    def __init__(self, manifest_path, subset, tokenizer, interference_speech, fps: int = 25):
        """
        manifest_path: path file JSONL
        Line: {"audio_filepath": "...", "video_filepath": "...", "text": "..."}
        """
        parts = []
        manifest_paths = [f.strip() for f in manifest_path.split(",") if f.strip()]
        for f in manifest_paths:
            part = load_dataset("json", data_files=f, split="train")
            print(f"  loaded {f}: {len(part)} rows")
            parts.append(part)
        combined = concatenate_datasets(parts)
        combined = combined.filter(filter_fn, num_proc=50)
        self.data = combined
        self.audio_transform = AudioTransform(subset=subset, speech_dataset=interference_speech)
        self.tokenizer = tokenizer
        self.fps = fps
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item['audio']
        video_path = item.get('video', None)
        text = item['text']

        tokens = self.tokenizer.encode(text) 
        labels = torch.tensor(tokens, dtype=torch.int64)

        lip_frames = None
        if video_path:
            _, lip_frames, _ = load_video_frames(
                video_path,
                lip_crop_mode="auto", 
                fps=self.fps,
                subset="train"
            )

        sync_samples = int(round(lip_frames.shape[0] * AUDIO_SAMPLE_RATE / self.fps)) if lip_frames is not None else None
        
        audio_np = load_audio(
            audio_path,
            sample_rate=AUDIO_SAMPLE_RATE,
            fallback_num_samples=sync_samples
        )
        audio = torch.tensor(audio_np, dtype=torch.float32)
        audio = self.audio_transform(audio.unsqueeze(1)).squeeze(1)
        return {
            "audio": audio,
            "video": lip_frames,
            "labels": labels
        }

def avasr_collate_fn(batch):
    batch_size = len(batch)
    audios = [item["audio"] for item in batch]
    videos = [item["video"] for item in batch] 
    labels = [item["labels"] for item in batch]

    audio_lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.int64)
    label_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.int64)
        
    padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    valid_videos = [v for v in videos if v is not None]

    if len(valid_videos) == 0:
        padded_videos = None
        video_lengths = None
    else:
        
        video_lengths = torch.tensor([v.shape[0] if v is not None else 0 for v in videos], dtype=torch.int64)
        tail_shape = valid_videos[0].shape[1:] 
        max_v_len = video_lengths.max().item()
        
        padded_shape = (batch_size, max_v_len, *tail_shape)
        padded_videos = torch.zeros(padded_shape, dtype=valid_videos[0].dtype)
        
        for i, v in enumerate(videos):
            if v is not None:
                t = v.shape[0]
                padded_videos[i, :t, ...] = v

    return {
        "audio_signal": padded_audios,
        "audio_lengths": audio_lengths,
        "video_frames": padded_videos,
        "video_lengths": video_lengths,
        "labels": padded_labels,
        "label_lengths": label_lengths
    }