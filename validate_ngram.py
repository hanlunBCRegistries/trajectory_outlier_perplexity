import os
import json
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import glob

def load_synthetic_outliers():
    """Load all synthetic outliers from the outliers directory"""
    outlier_trajectories = set()
    outlier_files = glob.glob("./data/porto/outliers/*.csv")
    
    print(f"Loading synthetic outliers from {len(outlier_files)} files...")
    for file_path in outlier_files:
        outlier_type = os.path.basename(file_path).split('_')[0]  # 'route_switch' or 'detour'
        with open(file_path) as f:
            for line in f:
                try:
                    traj = tuple(ast.literal_eval(line.strip()))
                    outlier_trajectories.add(traj)
                except Exception as e:
                    print(f"Error parsing outlier in {file_path}: {e}")
                    continue
    
    print(f"Loaded {len(outlier_trajectories)} synthetic outlier trajectories")
    return outlier_trajectories

def load_trajectories(filepath, max_trajectories=None):
    """Load trajectories from processed file"""
    trajectories = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if max_trajectories and i >= max_trajectories:
                break
            try:
                traj = ast.literal_eval(line.strip())
                trajectories.append(traj)
            except Exception as e:
                print(f"Error parsing line {i}: {e}")
                continue
    
    print(f"Loaded {len(trajectories)} trajectories")
    return trajectories

def main():
    # Load N-gram results
    with open("./results/ngram/porto/porto_ngram_3_results.json", 'r') as f:
        results = json.load(f)
    
    # Load all trajectories and synthetic outliers
    trajectories = load_trajectories("./data/porto/porto_processed.csv")
    synthetic_outliers = load_synthetic_outliers()
    
    # Get the indices of test set trajectories used by the model
    # We need to create train/test split the same way as in train_ngram.py
    np.random.seed(42)  # Must use the same seed as train_ngram.py
    indices = np.random.permutation(len(trajectories))
    split = int(0.8 * len(trajectories))
    test_idx = indices[split:]
    test_trajectories = [trajectories[i] for i in test_idx]
    
    # Count synthetic outliers in test set
    synthetic_in_test = sum(1 for traj_idx in test_idx if tuple(trajectories[traj_idx]) in synthetic_outliers)
    print(f"Synthetic outliers in test set: {synthetic_in_test} ({synthetic_in_test/len(test_idx)*100:.2f}%)")
    if synthetic_in_test == 0:
        print("WARNING: No synthetic outliers found in test set! Validation will not be meaningful.")
    
    # Get N-gram model outlier predictions (indices in the test set)
    model_outlier_idx = set(results['outlier_indices'])
    
    # Prepare ground truth labels and scores
    y_true = []
    y_scores = []
    
    for i, traj_idx in enumerate(test_idx):
        traj = trajectories[traj_idx]
        # Check if this trajectory is in the synthetic outliers set
        is_synthetic_outlier = tuple(traj) in synthetic_outliers
        # Check if this trajectory was detected as an outlier by the model
        is_detected_outlier = i in model_outlier_idx
        # Store ground truth (1 for outlier, 0 for normal)
        y_true.append(1 if is_synthetic_outlier else 0)
        # Store model score
        y_scores.append(results['scores'][i])
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Calculate metrics
    print("\n=== VALIDATION RESULTS ===")
    
    # Count of trajectories
    n_total = len(y_true)
    n_actual_outliers = sum(y_true)
    n_detected_outliers = len(model_outlier_idx)
    
    print(f"Test set size: {n_total} trajectories")
    print(f"Actual outliers: {n_actual_outliers} ({n_actual_outliers/n_total*100:.2f}%)")
    print(f"Detected outliers: {n_detected_outliers} ({n_detected_outliers/n_total*100:.2f}%)")
    
    # Create predictions using the model's threshold
    threshold = np.percentile(y_scores, results['threshold_percentile'])
    y_pred = (y_scores >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n=== PERFORMANCE METRICS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\n=== CONFUSION MATRIX ===")
    print(f"True Positives (correctly detected outliers): {tp}")
    print(f"False Positives (normal trajectories marked as outliers): {fp}")
    print(f"False Negatives (missed outliers): {fn}")
    print(f"True Negatives (correctly classified normal trajectories): {tn}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm_display = sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Normal', 'Outlier'], yticklabels=['Normal', 'Outlier'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./results/ngram/porto/confusion_matrix.png')
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)  # FIXED: Remove negation, higher is more anomalous
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('./results/ngram/porto/roc_curve.png')
    
    # Plot precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_curve, precision_curve)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('./results/ngram/porto/pr_curve.png')
    
    # Save detailed metrics to JSON
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn), 
        "true_negatives": int(tn),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "actual_outlier_count": int(n_actual_outliers),
        "detected_outlier_count": int(n_detected_outliers)
    }
    
    with open('./results/ngram/porto/validation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nDetailed metrics and plots saved to ./results/ngram/porto/")
    
    # Optional: Display a few examples of each category
    print("\n=== EXAMPLE TRAJECTORIES ===")
    
    def print_examples(description, condition, count=3):
        indices = np.where(condition)[0][:count]
        if len(indices) == 0:
            print(f"No examples found for: {description}")
            return
        
        print(f"\n{description}:")
        for idx in indices:
            test_traj_idx = test_idx[idx]
            print(f"  Trajectory {test_traj_idx}: {trajectories[test_traj_idx][:5]}... (Score: {y_scores[idx]:.4f})")
    
    # True Positives: Actual outliers correctly detected
    print_examples("TRUE POSITIVES (correctly detected outliers)", 
                  (y_true == 1) & (y_pred == 1))
    
    # False Positives: Normal trajectories incorrectly flagged as outliers
    print_examples("FALSE POSITIVES (normal trajectories incorrectly flagged)", 
                  (y_true == 0) & (y_pred == 1))
    
    # False Negatives: Missed outliers
    print_examples("FALSE NEGATIVES (missed outliers)", 
                  (y_true == 1) & (y_pred == 0))
    
    # Show plots
    plt.show()
    
    # Get scores for normal and outlier trajectories
    normal_scores = np.array([y_scores[i] for i in range(len(y_true)) if y_true[i] == 0])
    outlier_scores = np.array([y_scores[i] for i in range(len(y_true)) if y_true[i] == 1])

    print("\n=== SCORE STATISTICS ===")
    print(f"Normal trajectories: count={len(normal_scores)}, min={normal_scores.min():.4f}, mean={normal_scores.mean():.4f}, max={normal_scores.max():.4f}")
    print(f"Outlier trajectories: count={len(outlier_scores)}, min={outlier_scores.min():.4f}, mean={outlier_scores.mean():.4f}, max={outlier_scores.max():.4f}")

    # Create multiple visualization approaches
    plt.figure(figsize=(15, 10))

    # 1. Log scale histogram
    plt.subplot(2, 2, 1)
    plt.hist(normal_scores, bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(outlier_scores, bins=50, alpha=0.5, label='Outliers', density=True)
    plt.xscale('log')
    plt.legend()
    plt.title('Log Scale Distribution (Normalized)')
    plt.xlabel('Perplexity (Log Scale)')
    plt.ylabel('Density')

    # 2. Boxplot comparison
    plt.subplot(2, 2, 2)
    plt.boxplot([normal_scores, outlier_scores], labels=['Normal', 'Outliers'])
    plt.yscale('log')
    plt.title('Boxplot of Perplexity Scores (Log Scale)')
    plt.ylabel('Perplexity (Log Scale)')

    # 3. Zoomed view of higher perplexity range
    plt.subplot(2, 2, 3)
    # Find 99th percentile of all scores to set reasonable cutoff
    p99 = np.percentile(y_scores, 99)
    plt.hist(normal_scores[normal_scores < p99], bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(outlier_scores[outlier_scores < p99], bins=50, alpha=0.5, label='Outliers', density=True)
    plt.axvline(np.percentile(y_scores, 95), color='r', linestyle='--', label='95th percentile')
    plt.legend()
    plt.title(f'Distribution (Zoomed, <{p99:.1f} Perplexity)')
    plt.xlabel('Perplexity')
    plt.ylabel('Density')

    # 4. Cumulative distribution
    plt.subplot(2, 2, 4)
    # Sort scores and calculate cumulative proportions
    normal_sorted = np.sort(normal_scores)
    outlier_sorted = np.sort(outlier_scores)
    normal_y = np.arange(1, len(normal_sorted)+1) / len(normal_sorted)
    outlier_y = np.arange(1, len(outlier_sorted)+1) / len(outlier_sorted)

    plt.plot(normal_sorted, normal_y, label='Normal')
    plt.plot(outlier_sorted, outlier_y, label='Outliers')
    plt.axvline(np.percentile(y_scores, 95), color='r', linestyle='--', label='95th percentile')
    plt.xscale('log')
    plt.legend()
    plt.title('Cumulative Distribution (Log Scale)')
    plt.xlabel('Perplexity (Log Scale)')
    plt.ylabel('Cumulative Proportion')

    plt.tight_layout()
    plt.savefig('./results/ngram/porto/perplexity_distributions_multi.png')

if __name__ == "__main__":
    main()