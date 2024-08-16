import pandas as pd

def get_dataframe(path):
	df = pd.read_csv(path, sep='\t')
	return df

init_df = get_dataframe('zeit.predicted.vanilla1.tsv')
label_collection = []
confidence_collection = []

for i in list(range(1,11)):
	path = 'zeit.predicted.vanilla' + str(i) + '.tsv'
	df = get_dataframe(path)
	labels = df['predicted_labels']
	confidences = df['classifier_confidence']
	label_collection.append(labels)
	confidence_collection.append(confidences)

init_df = init_df.drop(['predicted_labels', 'classifier_confidence'], axis=1)

c = 0
for labels, confidences in zip(label_collection,confidence_collection):
	c += 1
	label_column_name = 'label_fold' + str(c)
	confidence_column_name = 'confidence_fold' + str(c)
	init_df[label_column_name] = labels
	init_df[confidence_column_name] = confidences

init_df.to_csv('zeit.predicted.vanilla.allfolds.tsv', sep='\t', index=False)
