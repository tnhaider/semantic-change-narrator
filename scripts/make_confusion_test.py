import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Collect tuples (label, predicted_label) from all files
labels = []
predicted_labels = []

for i in range(1, 11):
    filename = f"test.predicted.vanilla{i}.tsv"
    data = pd.read_csv(filename, sep='\t')
    labels.extend(data['label'])
    predicted_labels.extend(data['predicted_labels'])

# Create confusion matrix
cm = confusion_matrix(labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(labels)), yticklabels=sorted(set(labels)))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Test Vanilla Erzähler')

# Save the plot as a PNG file
plt.savefig('confusion_matrix_test_erzaehler.png')
plt.close()
