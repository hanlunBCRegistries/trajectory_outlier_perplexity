import time
import os
import math
import argparse
import json
from contextlib import nullcontext
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F

from datasets import PortoConfig, PortoDataset
from models.lstm_model import TrajectoryLSTM
from utils import seed_all, log, save_file_name_porto


def get_parser():
    """argparse arguments"""
    parser = argparse.ArgumentParser(description='Train LSTM for trajectory outlier detection')
    parser.add_argument('--data_dir', type=str, default='./data/porto')
    parser.add_argument('--data_file_name', type=str, default='porto_processed')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iters', type=int, default=20)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--threshold_percentile', type=float, default=95)
    parser.add_argument('--include_outliers', action='store_true', required=False)
    
    parser.add_argument('--out_dir', type=str, default='./results/lstm')
    parser.add_argument('--output_file_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1000)
    
    parser.add_argument('--compile', action='store_true', required=False)
    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    return args


@torch.no_grad()
def get_perplexity(model, data, device):
    """Calculate perplexity for trajectories"""
    model.eval()
    with torch.no_grad():
        inputs = data["data"].to(device)
        mask = data["mask"].to(device)
        
        # Get logits for each timestep
        logits = model(inputs[:, :-1])
        
        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract the log probability of the actual next tokens
        targets = inputs[:, 1:]
        next_token_log_probs = torch.gather(
            log_probs, 
            dim=2, 
            index=targets.unsqueeze(-1)
        ).squeeze(-1)
        
        # Zero out padding (where mask is 0)
        mask = mask[:, 1:]  # Shift mask to align with targets
        next_token_log_probs = next_token_log_probs * mask
        
        # Calculate log perplexity per sequence
        seq_lengths = mask.sum(dim=1)
        log_perplexity = -next_token_log_probs.sum(dim=1) / seq_lengths
        
    return log_perplexity


@torch.no_grad()
def validate_model(model, val_dataloader, device, ctx):
    """Run validation on the validation set"""
    model.eval()
    losses = []
    
    for data in val_dataloader:
        inputs = data["data"][:, :-1].contiguous().to(device)
        targets = data["data"][:, 1:].contiguous().to(device)
        
        with ctx:
            outputs = model(inputs)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)), 
                targets.view(-1),
                ignore_index=0  # Assuming 0 is the padding token
            )
        
        losses.append(loss.item())
    
    avg_val_loss = sum(losses) / len(losses)
    return avg_val_loss


def save_checkpoint(model, optimizer, epoch, batch_id, val_loss, args):
    """Save a model checkpoint"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'batch_id': batch_id,
        'val_loss': val_loss,
        'args': args,
        'dataset_config': args.dataset_config,
    }
    
    save_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}_batch_{batch_id}.pt")
    torch.save(checkpoint, save_path)
    log(f"Model checkpoint saved to {save_path}", args.log_file)


def detect_outliers(model, dataloader, device, threshold_percentile=95, ctx=nullcontext()):
    """Detect outliers based on perplexity scores"""
    log_perplexities = []
    metadata = []
    
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Calculating perplexities"):
            batch_perplexities = get_perplexity(model, data, device)
            log_perplexities.extend(batch_perplexities.tolist())
            metadata.extend(data["metadata"])
    
    # Determine threshold
    threshold = np.percentile(log_perplexities, threshold_percentile)
    
    # Find outliers
    outlier_indices = [i for i, perp in enumerate(log_perplexities) if perp > threshold]
    
    return outlier_indices, log_perplexities, threshold, metadata


def main():
    """Main training function"""
    args = get_parser()
    seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.out_dir, "train_log.txt")
    args.log_file = log_file
    with open(log_file, "w") as f:
        f.write("")  # Initialize empty log file
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
    ctx = nullcontext() if dtype == 'float32' else torch.amp.autocast(device_type=device.type, dtype=getattr(torch, dtype))
    log(f"Using device: {device}, dtype: {dtype}", args.log_file)
    if device.type == 'cuda':
        print("Running on GPU")
    else:
        print("Running on CPU")

    
    # Configure dataset
    dataset_config = PortoConfig(
        data_dir=args.data_dir,
        file_name=args.data_file_name,
        include_outliers=args.include_outliers,  # During training, normally we only use normal trajectories
    )
    args.dataset_config = dataset_config
    
    # Create dataset and dataloaders
    dataset = PortoDataset(dataset_config)
    train_indices, val_indices = dataset.partition_dataset()
    
    train_dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        collate_fn=dataset.collate, 
        sampler=SubsetRandomSampler(train_indices)
    )
    val_dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=dataset.collate, 
        sampler=SubsetRandomSampler(val_indices)
    )
    
    # Create model
    log("Creating LSTM model...", args.log_file)
    model = TrajectoryLSTM(
        vocab_size=len(dataset.dictionary),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Compile model if available and requested
    if args.compile and torch.__version__ >= "2.0.0":
        log("Compiling model...", args.log_file)
        model = torch.compile(model)
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(
        args.weight_decay,
        args.lr, 
        (args.beta1, args.beta2),
        device
    )
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # For early stopping
    save_model_count = 0
    
    # Start training
    log(f"Starting training for {args.max_iters} epochs", args.log_file)
    t0 = time.time()
    
    for epoch in range(args.max_iters):
        model.train()
        cumul_train_loss = 0
        cumulation = 0
        
        for batch_id, data in enumerate(train_dataloader):
            # Get inputs and targets (shift by 1 for next token prediction)
            inputs = data["data"].to(device)
            
            # Evaluation interval
            if batch_id % args.eval_interval == 0:
                val_loss = validate_model(model, val_dataloader, device, ctx)
                val_losses.append(val_loss)
                
                print(f"Validation loss: {val_loss:.4f}")
                
                # Save if best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model_count += 1
                    
                    # Save checkpoint
                    checkpoint_path = os.path.join(args.out_dir, f"checkpoint_{epoch}_{batch_id}.pt")
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'epoch': epoch,
                        'batch_id': batch_id,
                        'args': args,
                        'dataset_config': args.dataset_config,
                    }, checkpoint_path)
                    
                    log(f"Saved checkpoint to {checkpoint_path} (val_loss: {val_loss:.4f})", args.log_file)
                
                model.train()
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with ctx:
                outputs = model(inputs[:, :-1])
                targets = inputs[:, 1:]
                # Reshape for cross entropy
                loss = F.cross_entropy(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1),
                    ignore_index=dataset.dictionary.pad_token()
                )
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            
            # Log progress
            cumul_train_loss += loss.item()
            cumulation += 1
            
            if batch_id % args.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                log(f"Epoch {epoch+1}/{args.max_iters}, Batch {batch_id}/{len(train_dataloader)}: loss {loss.item():.4f}, time {dt*1000:.2f}ms", args.log_file)
        
        # Record average training loss for this epoch
        avg_epoch_loss = cumul_train_loss / max(1, cumulation)
        train_losses.append(avg_epoch_loss)
        log(f"Epoch {epoch+1} completed, Avg Train Loss: {avg_epoch_loss:.4f}", args.log_file)
    
    train_df = pd.DataFrame({"loss": train_losses, "type": "train"})
    val_df = pd.DataFrame({"loss": val_losses, "type": "validation", 
                        "step": [i * args.eval_interval for i in range(len(val_losses))]})
    train_df.to_csv(f"{args.out_dir}/train_losses_{args.output_file_name or 'final'}.tsv", sep="\t")
    val_df.to_csv(f"{args.out_dir}/val_losses_{args.output_file_name or 'final'}.tsv", sep="\t")
    
    # Save final model
    final_path = os.path.join(args.out_dir, f"final_model_{args.output_file_name or 'final'}.pt")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'args': args,
        'dataset_config': args.dataset_config,
    }, final_path)
    log(f"Final model saved to {final_path}", args.log_file)
    
    # Optional: Run outlier detection if in debug mode
    if args.debug:
        log("Running outlier detection on validation set", args.log_file)
        outliers, perplexities, threshold, meta = detect_outliers(
            model, val_dataloader, device, args.threshold_percentile, ctx
        )
        log(f"Found {len(outliers)} potential outliers in validation set", args.log_file)
        
        # Save outlier results
        results = {
            'outlier_indices': outliers,
            'perplexities': perplexities,
            'threshold': float(threshold),
            'threshold_percentile': args.threshold_percentile
        }
        with open(f"{args.out_dir}/outlier_detection_results.json", 'w') as f:
            json.dump(results, f)
    
    log("Training completed", args.log_file)


if __name__ == "__main__":
    main()