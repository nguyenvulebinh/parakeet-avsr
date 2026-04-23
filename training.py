import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torchaudio
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import sentencepiece as spm
from transformers import get_cosine_schedule_with_warmup
from transformers.utils.hub import cached_file

from dataset import AVASRDataset, DistributedSortedBatchSampler, avasr_collate_fn
from src.avasr.core.modeling import AVASRForTranscription
from src.avasr.vision.visual_features import get_visual_feats
from src.avasr.core.configuration import AVASRConfig
from src.avasr.metric.metrics import compute_wer

FPS = 25
DEFAULT_MODEL_ID = "nguyenvulebinh/parakeet-avsr"


def transfer_weights_with_expansion(pretrained_model, model):
    with torch.no_grad():
        pretrained_dict = pretrained_model.state_dict()
        
        for name, param_2 in model.named_parameters():
            if name in pretrained_dict:
                param_1 = pretrained_dict[name]
                
                if param_1.shape == param_2.shape:
                    param_2.copy_(param_1)
                
                # 2. Extend Linear Layer Weight [1030, 640] -> [1448, 640]
                elif len(param_1.shape) == 2 and param_1.shape[0] == 1030 and param_1.shape[1] == 640:
                    param_2[:1030, :].copy_(param_1)
                
                # 3. Extend Linear Layer Bias [1030] -> [1448]
                elif len(param_1.shape) == 1 and param_1.shape[0] == 1030:
                    param_2[:1030].copy_(param_1)
                
                # 4. Extend Embedding Layer [1025, 640] -> [1443, 640]
                elif len(param_1.shape) == 2 and param_1.shape[0] == 1025 and param_1.shape[1] == 640:
                    param_2[:1024, :].copy_(param_1[:1024, :])
                    # B. Copy Padding Token:  1024 -> 1442
                    param_2[1442, :].copy_(param_1[1024, :])

                    
    print("Hoàn tất copy weights!")
    return model

class AVASRTrainerModule(nn.Module):
    """
   
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio_signal, audio_lengths, video_frames, video_lengths, labels, label_lengths, num_speakers):
        visual_embeds = None
        if video_frames is not None:
            with torch.no_grad(): 
                visual_embeds = get_visual_feats(
                    self.model.vis_feat_extractor,
                    video_frames,
                    video_lengths,
                    num_speakers=num_speakers,
                    extract_all_layers=self.model.config.extract_visual_features_all_layers,
                    chunk_length=self.model.config.visual_chunk_length_sec,
                )
        encoded, encoded_len = self.model(
            audio_signal=audio_signal,
            audio_lengths=audio_lengths,
            visual_embeds=visual_embeds,
            visual_embed_lengths=video_lengths,
            num_speakers=num_speakers,
        )
        encoded_T = encoded.transpose(1, 2)

        labels_in = torch.where(labels == -100, self.model.config.blank_id, labels)
        decoder_out, _, _ = self.model.decoder(targets=labels_in, target_length=label_lengths)
        decoder_out_U = decoder_out.transpose(1, 2)

        joint_out = self.model.joint.joint(f=encoded_T, g=decoder_out_U) 

        num_classes = self.model.config.vocab_size + 1
        logits = joint_out[:, :, :, :num_classes].contiguous()

        encoded_len = torch.clamp(encoded_len, max=logits.shape[1])
        label_lengths_clamped = torch.clamp(label_lengths, max=logits.shape[2] - 1)
        max_T = int(encoded_len.max().item())
        max_U = int(label_lengths_clamped.max().item())

        logits_trimmed = logits[:, :max_T, :max_U + 1, :].contiguous()
        labels_trimmed = labels[:, :max_U].contiguous()
        
        def test_loss(logits_trimmed, labels_trimmed, encoded_len, label_lengths_clamped, blank_id, index):
            i = index

            T_i = encoded_len[i]
            U_i = label_lengths_clamped[i]
            logits_i = logits_trimmed[i:i+1, :T_i, :U_i+1, :].contiguous()
            targets_i = labels_trimmed[i:i+1, :U_i].contiguous()


            return torchaudio.functional.rnnt_loss(
                logits=logits_i,
                targets=targets_i.to(torch.int32),
                logit_lengths=encoded_len[i:i+1].to(torch.int32),
                target_lengths=label_lengths_clamped[i:i+1].to(torch.int32),
                blank=blank_id,
                reduction="mean"
            )
         
        loss = torchaudio.functional.rnnt_loss(
            logits=logits_trimmed,
            targets=labels_trimmed.to(dtype=torch.int32), 
            logit_lengths=encoded_len.to(dtype=torch.int32),
            target_lengths=label_lengths_clamped.to(dtype=torch.int32),
            blank=self.model.config.blank_id,
            reduction="mean"
        )
 
        return loss

def evaluate(model, eval_loader, tokenizer, device, is_main_process):
    model.eval()
    all_preds = []
    all_refs = []

    pbar = eval_loader
    if is_main_process:
        pbar = tqdm(eval_loader, desc="Evaluating", unit="batch", leave=False)

    with torch.no_grad():
        for batch in pbar:
            audio_signal = batch["audio_signal"].to(device, non_blocking=True)
            audio_lengths = batch["audio_lengths"].to(device, non_blocking=True)
            video_frames = batch["video_frames"].to(device, non_blocking=True) if batch["video_frames"] is not None else None
            video_lengths = batch["video_lengths"].to(device, non_blocking=True) if batch["video_frames"] is not None else None
            labels = batch["labels"] 
            label_lengths = batch["label_lengths"]
            num_speakers = torch.ones(audio_signal.size(0), dtype=torch.long, device=device)

            # Trích xuất đặc trưng hình ảnh
            visual_embeds = None
            if video_frames is not None:
                visual_embeds = get_visual_feats(
                    model.vis_feat_extractor,
                    video_frames,
                    video_lengths,
                    num_speakers=num_speakers,
                    extract_all_layers=model.config.extract_visual_features_all_layers,
                    chunk_length=model.config.visual_chunk_length_sec,
                )

            encoded, encoded_len = model(
                audio_signal=audio_signal,
                audio_lengths=audio_lengths,
                visual_embeds=visual_embeds,
                visual_embed_lengths=video_lengths,
                num_speakers=num_speakers,
            )

            results = model.decode(encoded, encoded_len)
            
            for i, res in enumerate(results):
                # Pred text
                pred_text = tokenizer.decode(res.tokens)
                all_preds.append(pred_text)

                valid_labels = [token.item() for token in labels[i][:label_lengths[i]] if token.item() != -100]
                ref_text = tokenizer.decode(valid_labels)
                all_refs.append(ref_text)

    gathered_preds = [None for _ in range(dist.get_world_size())]
    gathered_refs = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_preds, all_preds)
    dist.all_gather_object(gathered_refs, all_refs)

    wer = 0.0
    if is_main_process:
        flat_preds = [item for sublist in gathered_preds for item in sublist]
        flat_refs = [item for sublist in gathered_refs for item in sublist]
        
        wer = compute_wer(flat_refs, flat_preds)
        print(f"\n[Eval] WER: {wer:.4f}")

    model.train()
    return wer

def load_dataset(args, tokenizer, interference_speech, subset="train"):
    if subset == "train":
        manifest_path = args.train_files
    else:
        manifest_path = args.eval_files
        
    dataset = AVASRDataset(
        manifest_path=manifest_path,
        subset=subset,
        tokenizer=tokenizer,
        interference_speech=interference_speech
    )
    
    lengths = dataset.data['duration'][:] if subset == "train" else None
    if subset == "train":
        batch_sampler = DistributedSortedBatchSampler(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,      
            drop_last=False,
            lengths=lengths    
        )
        loader = DataLoader(
            dataset, 
            batch_sampler=batch_sampler, 
            collate_fn=avasr_collate_fn, 
            num_workers=4,
            pin_memory=True
        )
    else:
        batch_sampler = DistributedSampler(
            dataset,
            shuffle=False,
            drop_last=False
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=batch_sampler,
            collate_fn=avasr_collate_fn,
            num_workers=4,
            pin_memory=True
        )
    return batch_sampler, loader

def load_augment(args, is_main_process):
    interference_speech = None
    if args.augment_av:
        import datasets
        if not is_main_process: dist.barrier() # Sync download
        
        interference_speech = datasets.load_dataset(
            "nguyenvulebinh/AVYT", 
            "lrs2", 
            cache_dir='/home/tbnguyen/workspaces/mcorec_baseline/data-bin/cache', 
            data_files='lrs2/lrs2-train-*.tar'
        ).remove_columns(['__key__', '__url__'])['train']
        
        if is_main_process: dist.barrier()
        if is_main_process: print(f"[augment] Interference speech dataset loaded.")
    
    return interference_speech

def main(args):
    # Init Distributed Process Group DDP
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    is_main_process = (local_rank == 0)
    
    if is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    cache_dir = Path(args.cache_dir).resolve()
    model_id_path = Path(args.model_id)
    
    # Init tokenizer
    tokenizer = spm.SentencePieceProcessor()
    if not bool(args.custom_tokenizer):
        tok_path = cached_file(args.model_id, "tokenizer.model", cache_dir=args.cache_dir)
        tokenizer.load(tok_path)
        model = AVASRForTranscription.from_pretrained(args.model_id)
    else:
        # Load custom Vietnamese tokenizer and expand weights
        tokenizer.load(args.custom_tokenizer_path)
        
        vocab_size = tokenizer.get_piece_size()
        
        pretrained_model = AVASRForTranscription.from_pretrained(args.model_id)
        config = AVASRConfig(
            avhubert_config=pretrained_model.config.avhubert_config,
            vocab_size=vocab_size,
            blank_id=vocab_size
        )
        model = AVASRForTranscription(config)
        model = transfer_weights_with_expansion(pretrained_model, model)


    model.to(device)
    model.train()
    model.joint.log_softmax = True

    if bool(args.freeze_vis_feat):
        for p in model.vis_feat_extractor.parameters():
            p.requires_grad = False
            
    print("Num training parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        
    # Load augemnt dataset 
    interference_speech = load_augment(args, is_main_process)
        
    #Load dataset
    train_batch_sampler, train_loader = load_dataset(args, tokenizer, interference_speech, subset="train")
    evak_batch_sampler, eval_loader = load_dataset(args, tokenizer, interference_speech, subset="val")
        
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_training_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
    
    train_module = AVASRTrainerModule(model).to(device)
    ddp_model = DDP(
        train_module, 
        device_ids=[local_rank], 
        output_device=local_rank, 
        find_unused_parameters=True
    )

    for epoch in range(args.epochs):
        train_batch_sampler.set_epoch(epoch)
        ddp_model.train()
        
        total_loss = 0.0
        
        pbar = train_loader
        if is_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="it")
            
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()

            inputs = {
                "audio_signal": batch["audio_signal"].to(device, non_blocking=True),
                "audio_lengths": batch["audio_lengths"].to(device, non_blocking=True),
                "labels": batch["labels"].to(device, non_blocking=True),
                "label_lengths": batch["label_lengths"].to(device, non_blocking=True),
                "video_frames": batch["video_frames"].to(device, non_blocking=True) if batch["video_frames"] is not None else None,
                "video_lengths": batch["video_lengths"].to(device, non_blocking=True) if batch["video_frames"] is not None else None,
                "num_speakers": torch.ones(batch["audio_signal"].size(0), dtype=torch.long, device=device)
            }
                
            loss = ddp_model(**inputs)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if is_main_process:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                wandb.log({
                    "train/loss": loss.item(),
                    "train/epoch": epoch + (step / len(train_loader)),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "global_step": epoch * len(train_loader) + step
                })
                
                
                if step % args.save_steps == 0 and step > 0:
                    checkpoint_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch+1}-step-{step}"
                    ddp_model.module.model.save_pretrained(checkpoint_dir)
                    print(f"Saved checkpoint to {checkpoint_dir}")
                    
            if step % args.eval_steps == 0 and step > 0:
                base_model = ddp_model.module.model
                wer = evaluate(base_model, eval_loader, tokenizer, device, is_main_process)
                
                if is_main_process:
                    wandb.log({
                        "eval/wer": wer,
                        "global_step": step
                    })
                
        if is_main_process:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch+1}"
            ddp_model.module.model.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")
        
    if is_main_process:
        wandb.finish()         
    dist.destroy_process_group()
    
def parse_args():
    script_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        default=DEFAULT_MODEL_ID,
        help="Identifier of the pretrained model (local path or HuggingFace model ID)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=script_dir / "model-bin",
        help="Directory to cache downloaded or processed model files"
    )
    parser.add_argument(
        "--augment_av",
        type=int,
        default=1,
        help="Whether to apply audio-visual data augmentation (1: enable, 0: disable)"
    )
    parser.add_argument(
        "--train_files",
        type=str,
        required=True,
        help="Path to training dataset manifest (JSON/JSONL format)"
    )
    parser.add_argument(
        "--eval_files",
        type=str,
        required=True,
        help="Path to evaluation/validation dataset manifest (JSON/JSONL format)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./avasr_en",
        help="Directory to save checkpoints, logs, and outputs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per training step"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Initial learning rate for optimizer"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=2000,
        help="Number of steps between saving model checkpoints"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=2000,
        help="Number of steps between eval model checkpoints"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Ratio of total steps used for learning rate warmup"
    )
    parser.add_argument(
        "--freeze-vis-feat",
        type=int,
        default=0,
        help="Freeze visual feature extractor during training (1: freeze, 0: trainable)"
    )
    parser.add_argument(
        "--custom-tokenizer",
        type=int,
        default=0,
        help="(1: train with new tokenizer, 0: train with old tokenizer)"
    )
    parser.add_argument(
        "--custom-tokenizer-path",
        type=str,
        help="Custom tokenizer path, extend vietnamese token"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="avasr-training",
        help="Weights & Biases project name for experiment tracking"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom run name for Weights & Biases (optional)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)