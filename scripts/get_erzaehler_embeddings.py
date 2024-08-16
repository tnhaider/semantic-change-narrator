import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load BERT model and tokenizer
#model_name = 'bert-base-german-cased'  # Replace with your BERT model name
model_name = 'saved_models/vanilla_1'  # Replace with your BERT model name
#model_name = 'saved_models/fic_1'  # Replace with your BERT model name
#model_name = 'redewiedergabe/bert-base-historical-german-rw-cased'  # Replace with your BERT model name
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Load TSV file
tsv_file = '../data/vanilla_1/all.tsv'  # Replace with your TSV file
data = pd.read_csv(tsv_file, delimiter='\t')
texts = data['text'].tolist()
labels = data['label'].tolist()

# Tokenize and obtain contextual embeddings
embeddings = []
c = 0
for text in texts:
    c += 1
    print(c, len(texts))
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    token_index = inputs['input_ids'][0].tolist().index(tokenizer.convert_tokens_to_ids('Erzähler'))
    token_embedding = outputs.last_hidden_state[0, token_index].numpy()
    embeddings.append(token_embedding)

np.savez('interpretation_german-fic-test.npz', embeddings=embeddings, labels=labels)

# Dimensionality reduction with t-SNE
#tsne = TSNE(n_components=2, random_state=42)
#embedded_vectors = tsne.fit_transform(embeddings)

# Plot t-SNE visualization
#plt.figure(figsize=(10, 8))
#for i, label in enumerate(labels):
#    plt.scatter(embedded_vectors[i, 0], embedded_vectors[i, 1], label=label)
#plt.legend()
#plt.title("t-SNE Visualization of 'Erzähler' Embeddings")
#plt.xlabel("Dimension 1")
#plt.ylabel("Dimension 2")
#plt.savefig('erzaehler_pretrained_embedding.png')


