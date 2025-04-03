import json

# Load the results
with open("./results/ngram/porto/porto_ngram_3_results.json", 'r') as f:
    results = json.load(f)

# Get statistics
print(f"Number of outliers: {len(results['outlier_indices'])}")
print(f"Threshold percentile: {results['threshold_percentile']}%")

# Get score distribution
import numpy as np
import matplotlib.pyplot as plt

scores = np.array(results['scores'])
plt.hist(scores, bins=50)
plt.axvline(np.percentile(scores, results['threshold_percentile']), color='r')
plt.xlabel('Perplexity Score')
plt.ylabel('Number of Trajectories')
plt.title('Distribution of Trajectory Perplexity')
plt.show()