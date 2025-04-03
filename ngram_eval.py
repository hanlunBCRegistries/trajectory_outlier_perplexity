import argparse
import os
import ast
import json
import fnmatch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from models.NGramModel import NGramModel

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/porto")
    parser.add_argument("--data_file_name", type=str, default="porto_processed")
    parser.add_argument("--output_dir", type=str, default="./results/ngram/eval")
    parser.add_argument("--threshold_percentile", type=float, default=95)
    parser.add_argument("--include_outliers", action="store_true")
    parser.add_argument("--outliers_dir", type=str, default="./data/porto/outliers")
    parser.add_argument("--outlier_filename_pattern", type=str, default="*.csv")
    return parser.parse_args()

def visualize_results(perplexities, true_labels, metrics, output_dir):
    """Create visualizations of perplexity distributions and PR curve"""
    os.makedirs(output_dir, exist_ok=True)
    perplexities = np.array(perplexities)
    true_labels = np.array(true_labels)
    threshold = metrics['threshold']

    # 1. Distribution Plot
    plt.figure(figsize=(12, 7))
    plt_normal = perplexities[true_labels == 0]
    plt_outlier = perplexities[true_labels == 1]

    mean_perp = perplexities.mean()
    std_perp = perplexities.std()
    if std_perp < 1e-4:
        x_min = mean_perp - 0.1
        x_max = mean_perp + 0.1
    else:
        x_min = max(perplexities.min(), mean_perp - 3 * std_perp)
        x_max = min(perplexities.max(), mean_perp + 3 * std_perp)
        x_min = min(x_min, threshold - 0.1*abs(threshold))
        x_max = max(x_max, threshold + 0.1*abs(threshold))

    bins = np.linspace(x_min, x_max, 50)
    plt.hist(plt_normal, bins=bins, alpha=0.6, label='Normal', color='blue')
    plt.hist(plt_outlier, bins=bins, alpha=0.6, label='Outlier', color='red')
    plt.axvline(x=threshold, color='black', linestyle='--', 
                label=f'Threshold ({threshold:.4f})')
    plt.title('Score Distribution')
    plt.xlabel('Log Score')
    plt.ylabel('Count')
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=300)
    plt.close()

    # 2. PR curve
    try:
        precisions, recalls, _ = precision_recall_curve(true_labels, perplexities)
        pr_auc = auc(recalls, precisions)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, label=f'PR-AUC: {metrics["pr_auc"]:.4f}')
        plt.scatter(metrics['recall'], metrics['precision'], color='red', s=50, zorder=5,
                    label=f'Op Point (F1: {metrics["f1"]:.4f})')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating PR curve: {e}")

def main():
    args = get_parser()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_file_path}")
    try:
        model = NGramModel.load(args.model_file_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process normal data
    print("\n--- Processing Normal Trajectories ---")
    normal_trajectories = []
    try:
        with open(os.path.join(args.data_dir, f"{args.data_file_name}.csv"), 'r') as f:
            for line in tqdm(f, desc="Loading normal data"):
                try:
                    traj = ast.literal_eval(line.strip())
                    normal_trajectories.append(traj)
                except:
                    continue
    except FileNotFoundError:
        print(f"Error: Normal data file not found")
        return

    print(f"Loaded {len(normal_trajectories)} normal trajectories")
    
    # Calculate normal scores
    _, normal_scores = model.detect_outliers(normal_trajectories, args.threshold_percentile)
    threshold = np.percentile(normal_scores, args.threshold_percentile)
    print(f"Calculated scores for normal trajectories")
    print(f"Threshold at {args.threshold_percentile}th percentile: {threshold:.4f}")

    if not args.include_outliers:
        print("\nEvaluation finished (only normal data processed)")
        return

    # Process outlier files
    print("\n--- Finding and Processing Outlier Files ---")
    try:
        all_files = os.listdir(args.outliers_dir)
        outlier_files = [f for f in all_files 
                        if fnmatch.fnmatch(f, args.outlier_filename_pattern) 
                        and f.endswith('.csv')]
        if not outlier_files:
            print(f"No matching outlier files found in {args.outliers_dir}")
            return
        print(f"Found {len(outlier_files)} outlier files to evaluate")
    except FileNotFoundError:
        print(f"Error: Outliers directory not found: {args.outliers_dir}")
        return

    all_results = {}
    for outlier_file in outlier_files:
        outlier_path = os.path.join(args.outliers_dir, outlier_file)
        print(f"\nProcessing: {outlier_file}")
        
        # Load outliers
        outlier_trajectories = []
        with open(outlier_path, 'r') as f:
            for line in f:
                try:
                    traj = ast.literal_eval(line.strip())
                    outlier_trajectories.append(traj)
                except:
                    continue

        if not outlier_trajectories:
            print(f"No valid trajectories found in {outlier_file}")
            continue

        print(f"Loaded {len(outlier_trajectories)} outlier trajectories")

        # Calculate outlier scores
        _, outlier_scores = model.detect_outliers(outlier_trajectories, args.threshold_percentile)
        
        # Combine scores and create labels
        combined_scores = np.concatenate([normal_scores, outlier_scores])
        true_labels = np.concatenate([
            np.zeros(len(normal_scores)),
            np.ones(len(outlier_scores))
        ])
        
        # Calculate metrics
        predictions = (combined_scores > threshold).astype(int)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        precisions, recalls, _ = precision_recall_curve(true_labels, combined_scores)
        pr_auc = auc(recalls, precisions)

        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'pr_auc': float(pr_auc),
            'threshold': float(threshold)
        }
        
        # Store results
        all_results[outlier_file] = metrics
        
        # Create visualizations
        output_subdir = os.path.join(args.output_dir, os.path.splitext(outlier_file)[0])
        visualize_results(combined_scores, true_labels, metrics, output_subdir)
        
        # Print results
        print(f"Results for {outlier_file}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")

    # Save all results
    results_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll results saved to {results_path}")

if __name__ == "__main__":
    main()
