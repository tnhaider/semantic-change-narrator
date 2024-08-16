import sys, re
import pickle
import pandas
from somajo import SoMaJo
import json
from collections import Counter

tokenizer = SoMaJo("de_CMC", split_camel_case=True)


infile1 = open(sys.argv[1], 'rb')
#infile2 = open(sys.argv[2], 'rb')
#infile3 = open(sys.argv[3], 'rb')
df1 = pandas.read_csv(infile1, delimiter='\t')
#df1 = pickle.load(infile1)
#df2 = pickle.load(infile2)
#df3 = pickle.load(infile3)
infile1.close()
#infile2.close()
#infile3.close()

print(df1.info())
#print(df2.info())
#print(df3.info())

#df = pandas.concat([df1, df2, df3])
df = df1

#infile = open(sys.argv[1], 'rb')
#df = pickle.load(infile)
#infile.close()

#print(df.info())

linesyllablesep = "¬"
#header = ['filename', 'text', 'label']

filenames = df['filename']
texts = df['text']
#labels = df['label']
labels = ['Erzähler'] * len(texts)

#print(Counter(labels))


rows = zip(filenames, texts, labels)
years = []
lendocs = len(filenames)
newcorpus = {}
alltokens = []
idx = 0
indices = []

header = '\t'.join(['idx', 'token', 'context', 'sentence', 'label', 'year', 'quarter', 'filename'])
#header = '\t'.join(['idx', 'token', 'label', 'year', 'quarter', 'sentence', 'context', 'filename'])
#header = '\t'.join(['token', 'label', 'year', 'quarter', 'sentence', 'context', 'filename'])
outfile = open('erzähler_context_unlabeled.tsv', 'w')
outfile.write(header)
outfile.write('\n')

instances = 0
c = 0
for filename, text, label in rows:
	#if instances > 5:
	#	break
	print(idx, lendocs, instances)
	quarter = filename[:6]
	year = filename[:4]
	try:
		text = re.sub(linesyllablesep, '-', text)
	except TypeError:
		continue
	text = re.sub('-\s', '', text)
	text = text.strip()
	sentences = tokenizer.tokenize_text([text])
	sentences = list(sentences)
	sidx = 0
	for sentence in sentences:
		#print(idx, lendocs, len(indices), s, len(sentences))
		for token in sentence:
			ttl = token.text.lower()
			if re.match('erz[a-zA-ZäöüÄÖÜß. -]{1,2}.l.r.?.?$', ttl):
			#if re.match('schriftsteller.?.?$', ttl):
			#if re.match('chara.t.r.?.?$', ttl):
			#if re.match('fig.r.?.?$', ttl):
			#if ttl == 'erzähler' or ttl == 'erzahler' or ttl == 'erzaehler':
				#print(year, token.text, [token.text for token in sentence])
				#indices.append([label, filename, year, quarter, token.text, s, sentences, text])
				sentence = ' '.join([token.text for token in sentences[sidx]])
				try:

					contextpre2 = ' '.join([token.text for token in sentences[sidx-2]])
					contextpre1 = ' '.join([token.text for token in sentences[sidx-1]])
					contextpost1 = ' '.join([token.text for token in sentences[sidx+1]])
					contextpost2 = ' '.join([token.text for token in sentences[sidx+2]])
				except IndexError:
					#continue
					try:
						contextpre1 = ' '.join([token.text for token in sentences[sidx+1]])
						contextpre2 = ' '
						contextpost1 = ' '.join([token.text for token in sentences[sidx+2]])
						contextpost2 = ' '
					except IndexError:
						continue
				context = contextpre2 + ' ' + contextpre1 + ' ' + sentence + ' ' + contextpost1 + ' ' + contextpost2

				c += 1
				row = '\t'.join(['erzähler' + str(c), token.text, context, sentence, label, year, quarter, filename])

				#row = '\t'.join([str(c), token.text, label, year, quarter, sentence, context, filename])
				outfile.write(row)
				instances += 1
				outfile.write('\n')
		sidx += 1
	idx += 1
	#tokens = text.split()
	#alltokens += tokens
	#sentences = tokenizer.tokenize_text([text])
	#for sentence in sentences:
	#	for token in sentence:
	#print("{}\t{}\t{}".format(token.text, token.token_class, token.extra_info))
	#		tokens.append(token.text)
	#newcorpus[idx] = {'filename':filename, 'year':year, #'text':text, 
	#		'tokens':tokens}

#print(indices)
#for idc in indices:
#	label = idc[0]
#	filename = idc[1]
#	year = idc[2]
#	quarter = idc[3]
#	token = idc[4]
#	sidx = idc[5]
#	sentences = idc[6]
#	#print(sidx, len(sentences))
#	#text = idc[7]
#	#sentence = sentences[sidx]
#	#sentence = sentences[0]
#	#print(sentence)



#print(Counter(alltokens))
#newjsonfile = open('recension.corpus.gold_label.json', 'w')
#json.dump(newcorpus, newjsonfile)

#c = 0
#for item in df:
	#c += 1
	#if c>3:
	#	break
#	print(item)


#for item in df['text']:
	#c += 1
	#if c>3:
	#	break
#	print(item)

#c = 0
#for item in df['filename']:
#	c += 1
#	if c>10:
#		break
#	print(item)
