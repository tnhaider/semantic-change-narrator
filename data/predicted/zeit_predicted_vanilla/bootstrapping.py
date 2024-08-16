import pandas as pd
import numpy as np
from collections import Counter

# Load the data
df = pd.read_csv('zeit.predicted.vanilla.allfolds.tsv', sep='\t')

# Extract label columns
label_columns = [f'label_fold{i}' for i in range(1, 11)]

# Function to determine the majority label using bootstrapping
def bootstrapped_majority_label(labels, sample_size, n_iterations=1000):
    majority_counts = Counter()
    for _ in range(n_iterations):
        sampled_labels = np.random.choice(labels, sample_size, replace=True)
        most_common_label = Counter(sampled_labels).most_common(1)[0][0]
        majority_counts[most_common_label] += 1
    final_majority_label = majority_counts.most_common(1)[0][0]
    return final_majority_label

# Apply the bootstrapping function with gradually increasing sample sizes
def apply_bootstrapping(row, max_sample_size=10, step=1, n_iterations=100):
    labels = row[label_columns]
    print(labels)
    results = {}
    for sample_size in range(1, max_sample_size + 1, step):
        majority_label = bootstrapped_majority_label(labels, sample_size, n_iterations)
        results[f'majority_label_{sample_size}'] = majority_label
    return pd.Series(results)

# Apply the bootstrapping to each row
results_df = df.apply(lambda row: apply_bootstrapping(row), axis=1)

# Concatenate results with original DataFrame
df = pd.concat([df, results_df], axis=1)

# Save the result to a new CSV file
df.to_csv('bootstrapped_majority_labels.csv', sep='\t', index=False)

