import argparse
import os
import ast
import json
import numpy as np
from models.NGramModel import NGramModel

def main():
    parser = argparse.ArgumentParser(description='Train N-gram model for trajectory outlier detection')
    parser.add_argument('--dataset', type=str, default='porto', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--data_file_name', type=str, default='porto_processed', help='Data file name')
    parser.add_argument('--n', type=int, default=3, help='N-gram size')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Smoothing parameter')
    parser.add_argument('--out_dir', type=str, default='./results/ngram', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Load normal data
    print(f"Loading normal data from {args.data_dir}/{args.dataset}/{args.data_file_name}.csv")
    normal_trajectories = []
    with open(f"{args.data_dir}/{args.dataset}/{args.data_file_name}.csv", 'r') as f:
        for line in f:
            try:
                traj = ast.literal_eval(line.strip())
                normal_trajectories.append(traj)
            except:
                print(f"Error parsing line: {line}")
    
    print(f"Loaded {len(normal_trajectories)} normal trajectories")
    
    # Train model on all normal trajectories
    model = NGramModel(n=args.n, smoothing=args.smoothing)
    print(f"Training {args.n}-gram model on all {len(normal_trajectories)} trajectories...")
    model.fit(normal_trajectories)
    
    # Save model
    model_path = os.path.join(model_dir, f"{args.dataset}_ngram_{args.n}.pkl")
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()