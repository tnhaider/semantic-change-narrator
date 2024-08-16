import sys
import pandas as pd
import random
from random import shuffle
from collections import Counter
from sklearn.model_selection import train_test_split
import csv

#def write_set_to_file(X, y, filename, downsample=False):
#	f = open(filename, 'w')
#	header = '\t'.join(['text', 'label']) + '\n'
#	f.write(header)
#	for text, label in zip(X,y):
#		line = '\t'.join([text, label]) + '\n'
#		f.write(line)
#	print('Writing done ', filename)
#	print(Counter(y))

#def write_set_to_file(X, y, filename):
#    with open(filename, 'w', encoding='utf-8', newline='\n') as f:
#        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
#        writer.writerow(['text', 'label'])
#        for text, label in zip(X, y):
#            writer.writerow([text, label])
#    print(f'Writing done: {filename}')
#    print(Counter(y))

def write_set_to_file(X, y, filename):
    X = [' '.join(i.split()) for i in X]
    df = pd.DataFrame({'text': X, 'label': y})
    df.to_csv(filename, sep='\t', index=False)
    print(f'Writing done: {filename}')
    print(Counter(y))

df_raw = pd.read_csv(sys.argv[1], sep=';', encoding='utf8')
df_nox = df_raw[df_raw['Label_Gold'] != 'x']
df_aug = df_nox[df_nox['Zusatzinfo'] == 'augmented']
df = df_nox[df_nox['Zusatzinfo'] != 'augmented']


gold = [i[0] for i in df['Label_Gold']]
context = df['context']

data = list(zip(gold, context))
shuffle(data)

labels, texts = zip(*data)

random_number = random.randint(1, 300)

X_train, X_test_dev, y_train, y_test_dev = train_test_split(
    texts, labels, test_size=0.2, random_state=random_number)

X_test, X_dev, y_test, y_dev = train_test_split(
    X_test_dev, y_test_dev, test_size=0.5, random_state=random_number)

# Write Train
write_set_to_file(X_train, y_train, 'train.tsv')
# Write Test
write_set_to_file(X_test, y_test, 'test.tsv')
# Write Train
write_set_to_file(X_dev, y_dev, 'dev.tsv')


'''
l = []
r = []
c = 0
for i,j in data:
	s = i.split('/')
	cnt = Counter(l)
	#all_over_10 = all(value > 10 for value in cnt.values())
	#print(all_over_10)
	j = ' '.join(j.split())
	i = s[0]
	#if len(s) == 1:
	c += 1
	if c < 150 or cnt['c'] < 10:
		print(i)
		l.append(i)
		testfile.write('\t'.join([j,i]))
		testfile.write('\n')
	else:
		#if i == 'b' and Counter(r)['b'] > 300:
		#	continue
		trainfile.write('\t'.join([j,i]))
		trainfile.write('\n')
		r.append(i)

for i,j in data_aug:
	augmentedfile.write('\t'.join([j,i]))
	augmentedfile.write('\n')
		
print(Counter(l))
'''
