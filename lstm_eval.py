import os
import argparse
import json
import math
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
from tqdm import tqdm
import h3
import fnmatch
from argparse import Namespace
import torch.serialization

from models.lstm_model import TrajectoryLSTM
from datasets import PortoConfig, PortoDataset, VocabDictionary

# Add Namespace to safe globals
torch.serialization.add_safe_globals([Namespace])

def get_parser():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/porto")
    parser.add_argument("--data_file_name", type=str, default="porto_processed")
    parser.add_argument("--output_dir", type=str, default="./results/lstm/eval")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold_percentile", type=float, default=95)
    parser.add_argument("--include_outliers", action="store_true", 
                       help="Whether test data includes injected outliers")
    parser.add_argument("--outlier_level", type=int, default=3)
    parser.add_argument("--outlier_prob", type=float, default=0.1)
    parser.add_argument("--outlier_ratio", type=float, default=0.05)
    parser.add_argument("--outliers_dir", type=str, default="./data/porto/outliers",
                       help="Directory containing tokenized outlier CSV files. Overrides default derived from --data_dir.")
    parser.add_argument("--outlier_filename_pattern", type=str, default="*.csv",
                       help="Pattern to match specific outlier files (e.g., 'detour_*.csv', '*level_10*.csv').")
    
    args = parser.parse_args()
    return args


@torch.no_grad()
def get_perplexity(model, data, device, ctx, debug_print=False):
    """Calculate perplexity for each trajectory in batch"""
    # Move inputs to device
    inputs = data["data"].to(device)
    mask = data["mask"].to(device)
    
    if debug_print:
        print("\n--- Debug get_perplexity ---")
        print(f"Input shape: {inputs.shape}, Mask shape: {mask.shape}")
        # print(f"Sample input seq (first 10): {inputs[0, :10]}")

    with ctx:
        # Get logits for tokens except the last one
        logits = model(inputs[:, :-1])
        if debug_print:
            print(f"Logits shape: {logits.shape}")
            print(f"Logits stats: Min={logits.min():.4f}, Max={logits.max():.4f}, Mean={logits.mean():.4f}, Std={logits.std():.4f}")
            if not torch.all(torch.isfinite(logits)):
                 print("  WARNING: Non-finite values in logits!")

        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        if debug_print:
            print(f"LogProbs shape: {log_probs.shape}")
            print(f"LogProbs stats: Min={log_probs.min():.4f}, Max={log_probs.max():.4f}, Mean={log_probs.mean():.4f}, Std={log_probs.std():.4f}")
            if not torch.all(torch.isfinite(log_probs)):
                 print("  WARNING: Non-finite values in log_probs!")

        # Get targets (shifted by one position)
        targets = inputs[:, 1:]
        if debug_print:
             # print(f"Sample target seq (first 10): {targets[0, :10]}")
             pass # Targets are just indices

        # For each position, get the log probability of the actual next token
        target_indices = targets.unsqueeze(-1)
        next_token_log_probs = torch.gather(log_probs, dim=2, index=target_indices).squeeze(-1)
        if debug_print:
            print(f"NextTokenLogProbs shape: {next_token_log_probs.shape}")
            print(f"NextTokenLogProbs (before mask) stats: Min={next_token_log_probs.min():.4f}, Max={next_token_log_probs.max():.4f}, Mean={next_token_log_probs.mean():.4f}, Std={next_token_log_probs.std():.4f}")
            if not torch.all(torch.isfinite(next_token_log_probs)):
                 print("  WARNING: Non-finite values in next_token_log_probs (before mask)!")

        # Mask out padding tokens
        mask = mask[:, 1:]  # Shift mask to align with targets
        next_token_log_probs = next_token_log_probs * mask
        if debug_print:
            print(f"Mask (shifted) shape: {mask.shape}")
            # Print stats only for non-zero elements after masking for mean/std accuracy
            masked_probs = next_token_log_probs[mask.bool()]
            if masked_probs.numel() > 0:
                 print(f"NextTokenLogProbs (after mask, non-zero only) stats: Min={masked_probs.min():.4f}, Max={masked_probs.max():.4f}, Mean={masked_probs.mean():.4f}, Std={masked_probs.std():.4f}")
            else:
                 print("NextTokenLogProbs (after mask, non-zero only) stats: No non-masked elements!")

        # Calculate log perplexity per sequence
        seq_lengths = mask.sum(dim=1)
        # Add check for zero length sequences
        zero_len_mask = (seq_lengths == 0)
        if torch.any(zero_len_mask):
             print(f"  WARNING: {zero_len_mask.sum().item()} zero length sequence(s) detected after masking!")
             # Replace zero lengths with 1 to avoid NaN division, result for these seqs will be 0
             seq_lengths = seq_lengths.clamp(min=1)

        summed_neg_log_probs = -next_token_log_probs.sum(dim=1)
        log_perplexity = summed_neg_log_probs / seq_lengths 

        # Handle potential NaNs/Infs from division if clamping wasn't enough or probs were weird
        if not torch.all(torch.isfinite(log_perplexity)):
            print("  ERROR: Non-finite values detected in log_perplexity BEFORE correcting zero-length sequences!")
            # Replace NaNs/Infs perhaps with a very high value or zero?
            log_perplexity = torch.nan_to_num(log_perplexity, nan=0.0, posinf=0.0, neginf=0.0) # Replace with 0 for now

        # Ensure perplexity is 0 for sequences that had original length 0
        log_perplexity[zero_len_mask] = 0.0

        if debug_print:
            print(f"SeqLengths sample (first 5): {seq_lengths[:5]}")
            print(f"SummedNegLogProbs sample (first 5): {summed_neg_log_probs[:5]}")
            print(f"Final LogPerplexity shape: {log_perplexity.shape}")
            print(f"Final LogPerplexity stats: Min={log_perplexity.min():.4f}, Max={log_perplexity.max():.4f}, Mean={log_perplexity.mean():.4f}, Std={log_perplexity.std():.4f}")
            if not torch.all(torch.isfinite(log_perplexity)):
                 print("  ERROR: Non-finite values remain in final log_perplexity AFTER correction!")
            print("--- End Debug get_perplexity ---\n")
            
    return log_perplexity


@torch.no_grad()
def detect_outliers(model, dataloader, device, threshold_percentile=95, ctx=nullcontext()):
    """Detect outliers based on perplexity"""
    log_perplexities = []
    metadata = []
    
    model.eval()
    for data in tqdm(dataloader, desc="Calculating perplexities"):
        batch_perplexities = get_perplexity(model, data, device, ctx)
        log_perplexities.extend(batch_perplexities.cpu().tolist())
        metadata.extend(data["metadata"])
    
    # Determine threshold
    threshold = np.percentile(log_perplexities, threshold_percentile)
    
    # Find outliers
    outlier_indices = [i for i, perp in enumerate(log_perplexities) if perp > threshold]
    
    return outlier_indices, log_perplexities, threshold, metadata


def evaluate_outlier_detection(true_labels, predicted_scores, threshold_percentile=95):
    """Calculate precision, recall, F1-score, and PR AUC for outlier detection"""
    # Apply threshold to get binary predictions
    threshold = np.percentile(predicted_scores, threshold_percentile)
    predictions = (np.array(predicted_scores) > threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    # Calculate PR curve and AUC
    precisions, recalls, _ = precision_recall_curve(true_labels, predicted_scores)
    pr_auc = auc(recalls, precisions)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'threshold': threshold
    }


def visualize_results(perplexities, true_labels, metrics, output_dir):
    """Create visualizations of perplexity distributions and PR curve"""
    os.makedirs(output_dir, exist_ok=True)
    perplexities = np.array(perplexities)
    true_labels = np.array(true_labels)
    threshold = metrics['threshold']

    # 1. Zoomed-in Perplexity Distribution Histogram
    plt.figure(figsize=(12, 7))
    plt_normal = perplexities[true_labels == 0]
    plt_outlier = perplexities[true_labels == 1]

    # Calculate a narrow range around the mean/threshold for zooming
    mean_perp = perplexities.mean()
    std_perp = perplexities.std()
    # If std is very small, use a fixed small window, otherwise use a few std devs
    if std_perp < 1e-4:
        x_min = mean_perp - 0.1 # Adjust window size as needed
        x_max = mean_perp + 0.1
    else:
        x_min = max(perplexities.min(), mean_perp - 3 * std_perp)
        x_max = min(perplexities.max(), mean_perp + 3 * std_perp)
        # Ensure threshold is visible
        x_min = min(x_min, threshold - 0.1*abs(threshold))
        x_max = max(x_max, threshold + 0.1*abs(threshold))

    # Use a reasonable number of bins within the zoomed range
    bins = np.linspace(x_min, x_max, 50) 

    plt.hist(plt_normal, bins=bins, alpha=0.6, label='Normal', color='blue')
    plt.hist(plt_outlier, bins=bins, alpha=0.6, label='Outlier', color='red')
    plt.axvline(x=threshold, color='black', linestyle='--', 
                label=f'Threshold ({threshold:.4f})')
    plt.title('Zoomed-in Perplexity Distribution')
    plt.xlabel('Log Perplexity')
    plt.ylabel('Count')
    plt.xlim(x_min, x_max) # Apply the zoom
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "perplexity_distribution_zoomed.png"), dpi=300)
    plt.close()

    # 2. Scatter Plot with Jitter
    plt.figure(figsize=(12, 7))
    # Add small random vertical jitter for visualization
    y_jitter_normal = np.random.rand(len(plt_normal)) * 0.8 + 1.1 # Plot normal near y=1.5
    y_jitter_outlier = np.random.rand(len(plt_outlier)) * 0.8 + 0.1 # Plot outlier near y=0.5
    
    plt.scatter(plt_normal, y_jitter_normal, alpha=0.3, label='Normal', color='blue', s=10)
    plt.scatter(plt_outlier, y_jitter_outlier, alpha=0.3, label='Outlier', color='red', s=10)
    plt.axvline(x=threshold, color='black', linestyle='--', 
                label=f'Threshold ({threshold:.4f})')
    
    plt.title('Perplexity Scores (with Vertical Jitter)')
    plt.xlabel('Log Perplexity')
    plt.ylabel('Class (Visual Separation)')
    plt.ylim(0, 2) # Set y-limits for clarity
    plt.yticks([0.5, 1.5], ['Outlier', 'Normal']) # Label y-axis conceptually
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "perplexity_scatter_jitter.png"), dpi=300)
    plt.close()

    # 3. PR curve (Keep this as it might still show issues)
    try:
        precisions, recalls, _ = precision_recall_curve(true_labels, perplexities)
        pr_auc = auc(recalls, precisions)
    except ValueError as e:
        print(f"    Warning: Could not calculate PR curve: {e}. Might indicate degenerate scores.")
        pr_auc = float('nan') # Assign NaN if calculation fails
        # Create an empty plot or skip if AUC is NaN
        plt.figure(figsize=(10, 6))
        plt.title('Precision-Recall Curve (Calculation Error)')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True, alpha=0.3)
        plt.text(0.5, 0.5, 'Error calculating PR curve', horizontalalignment='center', verticalalignment='center')
        plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=300)
        plt.close()
        # Update metrics dict if needed, though evaluate_outlier_detection might have already handled this
        metrics['pr_auc'] = pr_auc 
        # metrics['precision'] = float('nan') # Mark other metrics potentially invalid too
        # metrics['recall'] = float('nan')
        # metrics['f1'] = float('nan')
        return # Exit visualization if PR curve fails fundamentally

    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, label=f'PR-AUC: {metrics.get("pr_auc", pr_auc):.4f}') # Use calculated or metric value
    # Plot the specific operating point from the threshold used
    if 'recall' in metrics and 'precision' in metrics:
        plt.scatter(metrics['recall'], metrics['precision'], color='red', s=50, zorder=5,
                    label=f'Op Point (F1: {metrics.get("f1", 0):.4f})')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=300)
    plt.close()


# --- Helper function to convert H3 sequence to coordinates ---
def h3_sequence_to_coords(h3_sequence):
    """Converts a sequence of H3 indices (expected as strings) to (lon, lat) coordinates."""
    coords = []
    for cell in h3_sequence:
        try:
            # Check if the item is a string and a valid H3 cell index
            if isinstance(cell, str) and h3.is_valid_cell(cell):
                # Use the correct function: h3.cell_to_latlng()
                lat, lon = h3.cell_to_latlng(cell)
                # Plotting often expects (x, y) -> (lon, lat)
                coords.append((lon, lat))
            # else: # Optional: Log if the cell is not a valid string H3
            #    if not (isinstance(cell, int) and cell == 0): # Ignore padding zeros
            #         print(f"Warning: Skipping invalid or non-string cell: {cell} (Type: {type(cell)})")
        except Exception as e:
            # Log unexpected errors during conversion
            print(f"Warning: Error converting H3 cell '{cell}': {e}")
            pass # Continue to the next cell
    return coords
# --- End Helper function ---

# --- Plotting function for misclassified trajectories ---
def visualize_misclassified(dataset, fp_indices, fn_indices, output_dir, max_plots=5):
    """Visualizes a sample of misclassified trajectories."""
    os.makedirs(output_dir, exist_ok=True)

    fp_coords_list = []
    print(f"Retrieving data for {min(len(fp_indices), max_plots)} False Positive trajectories...")
    for i in fp_indices[:max_plots]:
        # Access the original trajectory data from dataset.data
        raw_traj = dataset.data[i]
        # --- Debug Print Raw H3 ---
        print(f"  Raw FP traj {i} (len {len(raw_traj)}): Start={raw_traj[:3]}, End={raw_traj[-3:]}")
        # --- End Debug Print ---
        coords = h3_sequence_to_coords(raw_traj)
        if coords:
            fp_coords_list.append(coords)

    fn_coords_list = []
    print(f"Retrieving data for {min(len(fn_indices), max_plots)} False Negative trajectories...")
    for i in fn_indices[:max_plots]:
        raw_traj = dataset.data[i]
        # --- Debug Print Raw H3 ---
        print(f"  Raw FN traj {i} (len {len(raw_traj)}): Start={raw_traj[:3]}, End={raw_traj[-3:]}")
        # --- End Debug Print ---
        coords = h3_sequence_to_coords(raw_traj)
        if coords:
            fn_coords_list.append(coords)

    if not fp_coords_list and not fn_coords_list:
        print("No valid coordinates found for misclassified trajectories to plot.")
        return

    plt.figure(figsize=(12, 10))
    plot_made = False

    # Plot False Positives (Normal predicted as Outlier)
    for i, coords in enumerate(fp_coords_list):
        if len(coords) < 2: continue # Need at least 2 points to plot a line
        # --- Debug Print ---
        print(f"  Plotting FP trajectory {i} with {len(coords)} points.")
        if len(coords) >= 1:
            print(f"    Start: {coords[0]}, End: {coords[-1]}")
        # --- End Debug Print ---
        lons, lats = zip(*coords)
        plt.plot(lons, lats, marker='o', markersize=3, linestyle='-', color='orange', alpha=0.6, label='False Positive' if not plot_made else None)
        plt.scatter(lons[0], lats[0], color='green', s=30, zorder=5) # Start point
        plt.scatter(lons[-1], lats[-1], color='red', s=30, zorder=5)  # End point
        plot_made = True

    # Plot False Negatives (Outlier predicted as Normal)
    plot_made_fn = False # Use separate flag if mixing colors
    for i, coords in enumerate(fn_coords_list):
        if len(coords) < 2: continue
        # --- Debug Print ---
        print(f"  Plotting FN trajectory {i} with {len(coords)} points.")
        if len(coords) >= 1:
            print(f"    Start: {coords[0]}, End: {coords[-1]}")
        # --- End Debug Print ---
        lons, lats = zip(*coords)
        plt.plot(lons, lats, marker='x', markersize=4, linestyle='--', color='purple', alpha=0.6, label='False Negative' if not plot_made_fn else None)
        plt.scatter(lons[0], lats[0], color='blue', s=30, zorder=5) # Start point
        plt.scatter(lons[-1], lats[-1], color='black', s=30, zorder=5) # End point
        plot_made_fn = True
        plot_made = True # Mark that some plot was made

    if not plot_made:
        print("No trajectories with sufficient points found for plotting.")
        plt.close() # Close the empty figure
        return

    plt.title(f'Sample Misclassified Trajectories (Max {max_plots} each)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Auto-scale axes based on plotted data
    plt.axis('equal') # Make aspect ratio equal if desired
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "misclassified_trajectories.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved misclassified trajectory plot to {plot_path}")
    plt.close() # Close the figure after saving
# --- End Plotting function ---

# --- Function to load a single outlier file ---
def load_single_outlier_file(filepath):
    """Loads trajectories from a single outlier CSV file."""
    outlier_trajectories = []
    try:
        print(f"  Loading outliers from: {filepath}")
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    # Assuming each line is a list representation
                    traj = eval(line.strip())
                    outlier_trajectories.append(traj)
                except Exception as e_eval:
                    print(f"    Warning: Could not evaluate line in {filepath}: {line.strip()} - Error: {e_eval}")
        print(f"    Loaded {len(outlier_trajectories)} trajectories.")
    except FileNotFoundError:
        print(f"    Error: Outlier file not found at {filepath}")
        return None # Return None if file not found
    except Exception as e:
        print(f"    Error reading outlier file {filepath}: {e}")
        return None # Return None on other errors
    return outlier_trajectories
# --- End loading function ---

def main():
    """Main evaluation function"""
    args = get_parser()

    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
    ctx = nullcontext() if dtype == 'float32' else torch.amp.autocast(device_type=device.type, dtype=getattr(torch, dtype))
    print(f"Using device: {device}, dtype: {dtype}")

    # --- Load Model ---
    print(f"Loading model from {args.model_file_path}")
    try:
        checkpoint = torch.load(args.model_file_path, weights_only=False)  # Less secure but simpler option
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_file_path}")
        return
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return
    
    # --- Load Normal Data Config & Dictionary ---
    # Use config from checkpoint if available, otherwise create default
    # Ensure include_outliers=False for loading base data
    base_dataset_config = checkpoint.get('dataset_config', 
                                   PortoConfig(data_dir=args.data_dir, file_name=args.data_file_name))
    base_dataset_config.include_outliers = False # Explicitly ensure no outliers loaded here
    
    try:
        # Need dictionary for model init and data processing
        dictionary = VocabDictionary(os.path.join(base_dataset_config.data_dir, "vocab.json"))
    except FileNotFoundError:
        print(f"Error: vocab.json not found in {base_dataset_config.data_dir}")
        return
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return

    # --- Initialize Model ---
    # Initialize model with parameters from checkpoint or defaults
    saved_args = checkpoint.get('args', None)
    if saved_args:
        model = TrajectoryLSTM(
            vocab_size=len(dictionary),
            embedding_dim=saved_args.embedding_dim,
            hidden_dim=saved_args.hidden_dim,
            n_layers=saved_args.n_layers,
            dropout=saved_args.dropout
        )
    else:
        # Fallback defaults if args not in checkpoint
        model = TrajectoryLSTM(vocab_size=len(dictionary))
    
    try:
        model.load_state_dict(checkpoint['model'])
    except KeyError:
         print("Error: 'model' key not found in checkpoint state_dict.")
         return
    except Exception as e:
         print(f"Error loading model state_dict: {e}")
         return
         
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully.")

    # --- Process Normal Data ---
    print("\n--- Processing Normal Trajectories ---")
    try:
        normal_dataset = PortoDataset(base_dataset_config)
        normal_dataloader = DataLoader(
            normal_dataset, 
            batch_size=args.batch_size,
            collate_fn=normal_dataset.collate
        )
    except FileNotFoundError:
        print(f"Error: Normal data file not found at {os.path.join(base_dataset_config.data_dir, f'{base_dataset_config.file_name}.csv')}")
        return
    except Exception as e:
        print(f"Error creating normal dataset/dataloader: {e}")
        return

    normal_log_perplexities = []
    normal_metadata = [] # Keep metadata if needed (e.g., for detailed analysis)
    
    print_debug_normal = True # Flag to print debug info for first normal batch
    model.eval() # Ensure model is in eval mode
    for i, data in enumerate(tqdm(normal_dataloader, desc="Calculating perplexities for normal data")):
        debug_this_batch = print_debug_normal and i == 0
        with torch.no_grad(), ctx:
             batch_perplexities = get_perplexity(model, data, device, ctx, debug_print=debug_this_batch)
             normal_log_perplexities.extend(batch_perplexities.cpu().tolist())
             if 'metadata' in data:
                normal_metadata.extend(data["metadata"])
        if debug_this_batch:
             print_debug_normal = False # Turn off after first batch
             
    print(f"Processed {len(normal_log_perplexities)} normal trajectories.")
    if not normal_log_perplexities:
        print("Warning: No perplexities calculated for normal data.")
        return

    # --- Find and Loop Through Outlier Files ---
    if not args.include_outliers:
        print("\nEvaluation finished (only normal data processed as --include_outliers was not set).")
        # Optional: You could still analyze the normal perplexities here if desired
        # e.g., find threshold based only on normal data for anomaly detection
        # threshold = np.percentile(normal_log_perplexities, args.threshold_percentile)
        # outliers_indices = [i for i, p in enumerate(normal_log_perplexities) if p > threshold]
        # print(f"Found {len(outliers_indices)} potential anomalies within normal data using {args.threshold_percentile}th percentile threshold {threshold:.4f}")
        return

    print("\n--- Finding and Processing Outlier Files ---")
    # Determine outliers directory
    if args.outliers_dir is None:
        outliers_dir = os.path.join(args.data_dir, "outliers")
    else:
        outliers_dir = args.outliers_dir
    print(f"Looking for outliers in: {outliers_dir}")
    print(f"Using filename pattern: {args.outlier_filename_pattern}")

    try:
        all_files = os.listdir(outliers_dir)
        outlier_files = [f for f in all_files if fnmatch.fnmatch(f, args.outlier_filename_pattern) and f.endswith('.csv')]
        if not outlier_files:
            print(f"Warning: No matching outlier files found in {outliers_dir} with pattern '{args.outlier_filename_pattern}'")
            return
        outlier_filepaths = sorted([os.path.join(outliers_dir, f) for f in outlier_files])
        print(f"Found {len(outlier_filepaths)} matching outlier files to evaluate individually.")
    except FileNotFoundError:
        print(f"Error: Outliers directory not found: {outliers_dir}")
        return

    all_results = {} # Dictionary to store results per file

    print_debug_outlier = True # Flag to print debug info for first outlier batch overall
    # --- Loop for Individual Evaluation ---
    for outlier_filepath in outlier_filepaths:
        outlier_filename = os.path.basename(outlier_filepath)
        outlier_base_name = os.path.splitext(outlier_filename)[0]
        print(f"\n--- Evaluating against: {outlier_filename} ---")

        # 1. Load current outlier file data
        outlier_trajectories = load_single_outlier_file(outlier_filepath)
        if outlier_trajectories is None or not outlier_trajectories:
            print(f"  Skipping evaluation for {outlier_filename} due to loading error or empty file.")
            continue
        
        # 2. Prepare outlier data for model (needs collate_fn similar to PortoDataset)
        # Create a temporary dataset/dataloader for this specific outlier set
        # This requires adapting the collate function logic or creating a simple Dataset wrapper
        
        # --- Temporary Dataset/Collate for Outliers --- 
        # We need the dictionary from the normal dataset for encoding/padding
        class SingleOutlierSet(Dataset):
            def __init__(self, trajectories):
                self.trajectories = trajectories
                self.metadata = [outlier_base_name] * len(trajectories) # Label all as outlier type

            def __len__(self):
                return len(self.trajectories)

            def __getitem__(self, index):
                 sample = self.trajectories[index]
                 # Add SOT/EOT tokens as expected by collate/model
                 sample_with_ends = [dictionary.vocab["SOT"]] + sample + [dictionary.vocab["EOT"]] 
                 meta = self.metadata[index]
                 return sample_with_ends, meta

        def collate_outliers(data):
             masks = []
             token_lists = []
             metadatas = []
             # Find max length in this specific batch of outliers + SOT/EOT
             max_lenth = max([len(item[0]) for item in data]) 
             pad_token_id = dictionary.pad_token() 

             for tokens_, metadata in data: 
                 # Assuming tokens_ already has SOT/EOT and is list of H3 strings
                 encoded_tokens = dictionary.encode(tokens_)
                 mask = [1] * len(encoded_tokens) + [0] * (max_lenth - len(encoded_tokens))
                 padded_tokens = encoded_tokens + [pad_token_id] * (max_lenth - len(encoded_tokens))
                 token_lists.append(padded_tokens)
                 masks.append(mask)
                 metadatas.append(metadata)

             token_lists = torch.tensor(token_lists)
             masks = torch.tensor(masks)
             return {"data": token_lists, "mask": masks, "metadata": metadatas}
        # --- End Temp Dataset/Collate ---
        
        try:
            outlier_dataset_current = SingleOutlierSet(outlier_trajectories)
            outlier_dataloader_current = DataLoader(
                outlier_dataset_current,
                batch_size=args.batch_size, 
                collate_fn=collate_outliers
            )
        except Exception as e:
            print(f"  Error creating dataloader for {outlier_filename}: {e}")
            continue

        # 3. Calculate perplexities for this outlier set
        outlier_log_perplexities_current = []
        model.eval()
        for i, data in enumerate(tqdm(outlier_dataloader_current, desc=f"Calculating perplexities for {outlier_base_name}")):
            # Skip empty batches potentially returned by error in collate_fn
            if data['data'].numel() == 0: continue
            debug_this_batch = print_debug_outlier and i == 0 # Check overall flag and batch index
            with torch.no_grad(), ctx:
                try:
                    batch_perplexities = get_perplexity(model, data, device, ctx, debug_print=debug_this_batch)
                    outlier_log_perplexities_current.extend(batch_perplexities.cpu().tolist())
                except Exception as e_perp:
                     print(f"Error calculating perplexity for batch in {outlier_filename}: {e_perp}")
            if debug_this_batch:
                 print_debug_outlier = False # Turn off after first outlier batch overall

        if not outlier_log_perplexities_current:
            print(f"  No perplexities calculated for {outlier_filename}. Skipping evaluation.")
            continue
            
        # 4. Combine normal and current outlier perplexities & create labels
        combined_perplexities = normal_log_perplexities + outlier_log_perplexities_current
        true_labels = [0] * len(normal_log_perplexities) + [1] * len(outlier_log_perplexities_current)
        # Combine metadata if needed for plotting later (be mindful of length mismatch if any errors occurred)
        # combined_metadata = normal_metadata + [outlier_base_name] * len(outlier_log_perplexities_current)
        
        # --- Debug: Check Perplexity Stats ---
        norm_perp = np.array(normal_log_perplexities)
        out_perp = np.array(outlier_log_perplexities_current)
        print(f"    Normal Perplexity Stats: Min={norm_perp.min():.4f}, Max={norm_perp.max():.4f}, Mean={norm_perp.mean():.4f}, Std={norm_perp.std():.4f}")
        print(f"    Outlier Perplexity Stats: Min={out_perp.min():.4f}, Max={out_perp.max():.4f}, Mean={out_perp.mean():.4f}, Std={out_perp.std():.4f}")
        # Check for non-finite values
        if not np.all(np.isfinite(norm_perp)) or not np.all(np.isfinite(out_perp)):
             print(f"    ERROR: Non-finite perplexity values detected! Skipping evaluation for {outlier_filename}")
             continue
        # --- End Debug ---

        # 5. Evaluate (calculate threshold, metrics, visualize)
        print(f"  Evaluating {len(normal_log_perplexities)} normal vs {len(outlier_log_perplexities_current)} outliers from {outlier_filename}")
        try:
            metrics = evaluate_outlier_detection(true_labels, combined_perplexities, args.threshold_percentile)
            print(f"  Evaluation results for {outlier_filename}:")
            print(f"    Threshold (@{args.threshold_percentile}%): {metrics['threshold']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-score: {metrics['f1']:.4f}")
            print(f"    PR-AUC: {metrics['pr_auc']:.4f}")
            
            # Customize output dir/filenames for individual results
            individual_output_dir = os.path.join(args.output_dir, outlier_base_name)
            os.makedirs(individual_output_dir, exist_ok=True)
            
            # Visualize results specific to this outlier set
            visualize_results(combined_perplexities, true_labels, metrics, individual_output_dir)
            
            # Optional: Visualize misclassified for this specific run 
            # This requires getting the original trajectory data for this specific outlier set again
            # predictions = (np.array(combined_perplexities) > metrics['threshold']).astype(int)
            # fp_indices_combined = [i for i, (p, t) in enumerate(zip(predictions, true_labels)) if p==1 and t==0]
            # fn_indices_combined = [i for i, (p, t) in enumerate(zip(predictions, true_labels)) if p==0 and t==1]
            # Need to map fn_indices_combined back to indices within outlier_trajectories_current for plotting
            # fn_indices_outlier = [i - len(normal_log_perplexities) for i in fn_indices_combined]
            # visualize_misclassified_single(normal_dataset, fp_indices_combined, outlier_dataset_current, fn_indices_outlier, individual_output_dir)

            # Save metrics to JSON specific to this outlier set
            metrics_filepath = os.path.join(individual_output_dir, "metrics.json")
            with open(metrics_filepath, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Saved metrics and plots to: {individual_output_dir}")
            
            all_results[outlier_filename] = metrics # Store results
            
        except Exception as e_eval:
             print(f"  Error during evaluation for {outlier_filename}: {e_eval}")
             import traceback
             traceback.print_exc()

    # --- Summary --- (Optional)
    print("\n--- Evaluation Summary ---")
    if all_results:
        summary_df = pd.DataFrame.from_dict(all_results, orient='index')
        print(summary_df)
        summary_df.to_csv(os.path.join(args.output_dir, "evaluation_summary.csv"))
        print(f"\nSaved evaluation summary to {os.path.join(args.output_dir, 'evaluation_summary.csv')}")
    else:
        print("No outlier files were successfully evaluated.")

    print("\nEvaluation script finished.")


if __name__ == "__main__":
    main()