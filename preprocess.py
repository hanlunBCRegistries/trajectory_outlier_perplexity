# Python
"""
This code is based on the following github: https://github.com/liuyiding1993/ICDE2020_GMVSAE/blob/master/preprocess/preprocess.py
"""
import os
from collections import defaultdict
import json
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import h3 


def height2lat(height):
    return height / 110.574

def width2lng(width):
    return width / 111.320 / 0.99974

def in_boundary(lat, lng, b):
    return b['min_lng'] < lng < b['max_lng'] and b['min_lat'] < lat < b['max_lat']

def main(
        data_file_path,
        out_dir,
        grid_height,
        grid_width,
        boundary,
        min_traj_length,
        override=False,
):
    """
    Preprocess the proto dataset using H3 hexagonal indexing.
    """
    # Set H3 resolution (adjust based on desired cell size)
    h3_res = 9

    # Although grid_height and grid_width are still provided,
    # we now use H3 to compute hex tokens.
    # Change this line in the main function
    trajectories = pd.read_csv(data_file_path, header=0, index_col="TRIP_ID")
    processed_trajectories = defaultdict(list)

    shortest, longest = 20, 1200
    valid_trajectory_count = 0
    original_valid_count = 0 # Keep track before filtering stationary

    for i, (idx, traj) in enumerate(tqdm(trajectories.iterrows(), total=trajectories.shape[0], desc="process trajectories")):
        grid_seq = []
        valid = True
        try:
            polyline = eval(traj["POLYLINE"])
        except:
             # print(f"Warning: Could not eval polyline for index {idx}. Skipping.")
             continue # Skip this trajectory if polyline is invalid

        if shortest <= len(polyline) <= longest:
            for lng, lat in polyline:
                if in_boundary(lat, lng, boundary):
                    # Convert each (lat, lng) to its H3 hexagon index.
                    hex_token = h3.latlng_to_cell(lat, lng, h3_res)
                    grid_seq.append(hex_token)
                else:
                    valid = False
                    break

            if valid:
                original_valid_count += 1 # Count trajectories passing initial checks
                # --- Filter Stationary Trajectories ---
                # Calculate the ratio of unique points to total length
                if len(grid_seq) > 0: # Avoid division by zero for empty sequences
                    unique_ratio = len(set(grid_seq)) / len(grid_seq)
                else:
                    unique_ratio = 0

                # Define a threshold (e.g., require at least 10% unique points)
                # Adjust this threshold based on your data and needs
                min_unique_ratio = 0.1

                if unique_ratio >= min_unique_ratio:
                    # If trajectory has enough unique points, proceed
                    s, d = grid_seq[0], grid_seq[-1]
                    processed_trajectories[(s, d)].append(grid_seq)
                    valid_trajectory_count += 1
                # else: # Optional: Log discarded stationary trajectories
                #    print(f"Discarding stationary trajectory {idx} (unique ratio {unique_ratio:.2f}): {grid_seq[:5]}...")
                # --- End Filter ---

    print(f"Original valid trajectory num (length & boundary): {original_valid_count}")
    print(f"Valid (non-stationary) trajectory num (unique ratio >= {min_unique_ratio}): {valid_trajectory_count}")
    # For reference, you can still print the H3 resolution.
    print("Using H3 resolution:", h3_res)

    os.makedirs(out_dir, exist_ok=True)
    processed_file_path = f"{out_dir}/porto_processed.csv" # Define path
    fout = open(processed_file_path, "w")

    saved_trajectory_count = 0
    max_trajectory_size = 0
    print(f"Saving {valid_trajectory_count} valid & non-stationary trajectories...")
    # --- Loop to save trajectories ---
    for trajs in tqdm(processed_trajectories.values(), desc="save valid trajectories"):
        # Remove the incorrect filter: if len(trajs) >= min_traj_length:
        # Now save all trajectories that passed previous checks
        for traj in trajs:
            fout.write(f"{traj}\n")
            saved_trajectory_count += 1

            if len(traj) > max_trajectory_size:
                max_trajectory_size = len(traj)

    fout.close() # Ensure file is closed before proceeding
    print(f"Actual saved trajectory num: {saved_trajectory_count}")
    print(f"max trajectory size: {max_trajectory_size}")
    # --- End Saving Loop ---

    # --- Vocabulary Generation ---
    vocab_path = f"{out_dir}/vocab.json"
    if not os.path.exists(vocab_path) or override:
        print(f"Generating vocabulary file: {vocab_path}")
        vocab = {}
        next_id = 0

        # Assign fixed IDs to special tokens first
        for special_token in ["PAD", "SOT", "EOT"]:
            if special_token not in vocab:
                vocab[special_token] = next_id
                next_id += 1

        # Collect all unique H3 tokens from the processed data
        unique_h3_tokens = set()
        print("Collecting unique H3 tokens...")
        # Read the saved file (now guaranteed to be closed)
        try:
            with open(processed_file_path, "r") as f: # Use defined path
                 for line in tqdm(f, desc="Reading trajectories for vocab"):
                     # Assuming line is like "['token1', 'token2', ...]"
                     try:
                         traj_list = eval(line.strip())
                         unique_h3_tokens.update(traj_list)
                     except: # Handle potential errors if a line is malformed
                         print(f"Warning: Could not parse line for vocab: {line.strip()}")
        except FileNotFoundError:
             print(f"Error: Cannot find {processed_file_path} to build vocabulary.")
             # Exit or handle error - fallback is unreliable now
             return # Exit if file not found

        print(f"Found {len(unique_h3_tokens)} unique H3 tokens.")

        # Assign integer IDs to the unique H3 tokens
        for token in sorted(list(unique_h3_tokens)): # Sort for deterministic assignment
            if token not in vocab: # Check if it's not already a special token (unlikely for H3)
                vocab[token] = next_id
                next_id += 1

        print(f"Total vocabulary size: {len(vocab)}")
        # Ensure special tokens have the expected IDs (optional check)
        print(f"PAD ID: {vocab.get('PAD', 'Not Found')}")
        print(f"SOT ID: {vocab.get('SOT', 'Not Found')}")
        print(f"EOT ID: {vocab.get('EOT', 'Not Found')}")


        with open(vocab_path, "w", encoding="utf-8") as fp:
            json.dump(vocab, fp, indent=4) # Use indent for readability
        print(f"Vocabulary saved to {vocab_path}")

    # generate outliers (unchanged)
    from datasets import PortoConfig, PortoDataset

    """
    outlier_level: int = 3
    outlier_prob: float = 0.3
    outlier_ratio: float = 0.05
    """
    outlier_configs = [
        [0.05, 1, 0.2],   # Challenging outliers
        [0.05, 2, 0.3],   # Challenging outliers
        [0.05, 3, 0.4],   # Original mild outliers
    ]

    print("\nStarting outlier generation...")
    for outlier_config in outlier_configs:
        print(f"Generating outliers for config: ratio={outlier_config[0]}, level={outlier_config[1]}, prob={outlier_config[2]}")
        
        dataset_config = PortoConfig(
            data_dir=out_dir,
            file_name="porto_processed",
            outlier_ratio=outlier_config[0],
            outlier_level=outlier_config[1],
            outlier_prob=outlier_config[2],
            include_outliers=False
        )
        
        try:
            print(f"Initializing dataset from: {dataset_config.data_dir}/{dataset_config.file_name}.csv")
            # PortoDataset will open the processed_file_path
            dataset = PortoDataset(dataset_config)
            
            print(f"Dataset loaded with {len(dataset)} trajectories for outlier generation.")
            
            if len(dataset) > 0:
                 dataset.generate_outliers()
            else:
                 print("Warning: Dataset loaded 0 trajectories. Skipping outlier generation for this config.")
                 save_dir = f"{dataset_config.data_dir}/outliers"
                 os.makedirs(save_dir, exist_ok=True)
                 for key in ["route_switch", "detour", "sharp_turn"]:
                     current_save_dir = \
                         f"{save_dir}/{key}_ratio_{dataset_config.outlier_ratio}_level_{dataset_config.outlier_level}_prob_{dataset_config.outlier_prob}.csv"
                     print(f"Saving empty outlier file {current_save_dir}")
                     with open(current_save_dir, "w") as fout:
                         pass


        except FileNotFoundError:
             print(f"Error: Could not find processed file in {dataset_config.data_dir} to generate outliers.")
        except Exception as e:
            print(f"An error occurred during outlier generation for config {outlier_config}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--min_traj_length", type=int, default=25)
    parser.add_argument("--grid_height", type=float, default=0.1)  
    parser.add_argument("--grid_width", type=float, default=0.1)   
    parser.add_argument("--out_dir", type=str, default="./data/porto")

    args = parser.parse_args()

    BOUNDARY = {'min_lat': 41.140092, 'max_lat': 41.185969, 'min_lng': -8.690261, 'max_lng': -8.549155}

    main(
        data_file_path=args.data_dir,
        out_dir=args.out_dir,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        boundary=BOUNDARY,
        min_traj_length=args.min_traj_length,
        override=False
    )