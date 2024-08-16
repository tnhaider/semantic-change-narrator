# fmt: off
import logging
from pathlib import Path
import sys

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
import pandas as pd

version = sys.argv[1]
#df = pd.read_csv('../data/erzaehler_gold_' + version + '/test.tsv', sep='\t')
#df = pd.read_csv('zeit.stimuli.erzaehler.2sentencecontext.filteredbygold.shuffle.tsv', sep='\t')
df = pd.read_csv(sys.argv[2], sep='\t')
texts = df['text']
#labels = df['label']
#your_model_dir = 'saved_models/bert_model_erzaehler_balanced'
your_model_dir = 'saved_models/vanilla_' + version 
# down-stream inference
basic_texts = []
basic_texts2 = []
#    {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
#    {"text": "Martin Müller spielt Handball in Berlin"},
#]
c = 0
print('texts', len(texts))
for text in texts:
	c += 1
	if c < 45:
		basic_texts.append({'text':text})
	else:
		basic_texts2.append({'text':text})
#model = Inferencer.load(your_model_dir)
# LM embeddings
model = Inferencer.load(your_model_dir, extraction_strategy="cls_token", extraction_layer=-1, gpu=True, batch_size=1, num_processes=1)
#model.inference_from_dicts(dicts=basic_texts)
result = model.inference_from_dicts(dicts=basic_texts)
result2 = model.inference_from_dicts(dicts=basic_texts2)

#print(result)

contexts = []
labels = []
confidences = []
for i in result:
	info = i['predictions'][0]
	contexts.append(info['context'])
	labels.append(info['label'])
	confidences.append(info['probability'])
for i in result2:
	info = i['predictions'][0]
	contexts.append(info['context'])
	labels.append(info['label'])
	confidences.append(info['probability'])

model.close_multiprocessing_pool()

print(labels)
print(confidences)
print(len(labels))
print(len(confidences))
print(len(texts))

df['predicted_labels'] = labels
df['classifier_confidence'] = confidences
#df['predicted_texts'] = contexts

print(df)
df.to_csv('test.predicted.vanilla'+ version +'.tsv', sep='\t')

'''
    # 9. Load it & harvest your fruits (Inference)
    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
        {"text": "Martin Müller spielt Handball in Berlin"},
        {"text": "Der 1879 erschienene Band  Lustige Geschichten' allerdings sieht schon etwas darnach aus , als verdanke er seine Veröffentlichung der kleinen Schwäche , zu welcher die errungene Popularität verleitet : einmal geschriebenes nicht unterdrücken zu können . Sie sind wohl auf das Conto früherer Jahre zu schreiben und hätten nur zum geringen Theile einen nochmaligen Abdruck aus den Kalendern und Zeitschriften , in denen sie zuerst erschienen , verdient . Wie sich diese Skizzen oft novellistisch abrunden , so sind häufig die in den Novellenbänden abgedruckten breitern Sachen mehr skizzenhafte Episoden als abgerundete Erzählungen , oder eine Reihe von Einzelbildern wird lose durch einen novellistischen Faden verbunden , wie in den  Schriften des Waldschulmeisters' ( 1875 ) , denen wir trotz des leichten Gefüges , trotz der auch hier die Harmonie des Ganzen störenden , etwas unwahrscheinlichen und unnöthiger Romantik der persönlichen Geschichte des Helden oder vielmehr Erzählers die  Schriften' des Schulmeisters sind sein Tagebuch , welches sein Leben und Wirken in demi entlegenen Waldwinkel schildert , in den er sich vor der Welt geflüchtet hat  den Preis unter den Roseggerschen Schriften zuerkennen möchten . Daneben findet sich in den frühern und spätern Bänden anderes , was sich durch den Gang der dargestellten Ereignisse oder den Gedanken , welchen die Erzählung illustriren soll , fest gliedert und künstlerisch abrundet . Kampf und Streit ist das Loos aller Mensche » , sagt Rosegger im Vorworte zu  Streit und Sieg' ; der Kampf gegen das Elementare der äußern Natur und des heißen Blutes im Menschenherzen , der Kampf der Cultur gegen die rohe Gewalt , der Vernunft gegen die Beschränktheit , der Seeleuhoheit gegen den Egoismus , des Lebens gegen die Starrniß ."},
        {"text": "Ich erinnere mich noch , wie es mich manchmal verdroß , als ich sie allmählich das Übergewicht gewinnen sah . Auch jetzt gehöre ich nur bedingt zu Fritz Reuters Verehrern , sein Humor ist mir oft zu derb , sein Sentiment wird leicht rührselig , und ich stimme darin mit Vartels überein , daß er weder ein großer Poet noch ein großer Künstler ist . Aber als unterhaltender Erzähler hat er von Anfang an gewinnende Töne angeschlagen , und sein Plattdeutsch mag so echt oder so unecht sein , wie es will : es liest sich leichter als die Sprache Klaus Groths , die dem Oberdeutschen ohne Wörterbuch kaum verständlich ist . Übrigens haben die beiden außer dem , daß sie plattdeutsch schrieben , kaum noch etwas mit einander gemeinsam . Fritz Reuter mit seinem leichtern Gefährt gewann schnell einen deutlichen Vorsprung , viel verdankt er auch seinen Landsleuten , die ihm überall Quartier bereiteten ; kaum ging ein gebildeter junger Mecklenburger über die Grenze , der nicht einen Band Reuter in der Tasche hatte und daraus vorzulesen bereit war . plattdeutsch und Hochdeutsch Ganz gewiß ist Schleswig-Holstein geschichtlich und kulturhistorisch reicher und anziehender als die beideu Mecklenburg , und Dithmarschen oder Heide und Marne klingt und stimmt anders als Neubrandenburg und Stavenhagen ."}
    ]
    model = Inferencer.load(save_dir)
    result = model.inference_from_dicts(dicts=basic_texts)
    print(result)
    model.close_multiprocessing_pool()
'''
