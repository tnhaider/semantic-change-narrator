## Overview
This repository contains data and code to support the paper:

_"The Ongoing Birth of the Narrator: Empirical Evidence for the Emergence of the Author-Narrator Distinction in Literary Criticism"_

## Folders
- figures
- guidelines
  - contains the guidelines used during annotation
- data
  - all (all labeled instances and gold)
  - vanilla (folds 1-10 for vanilla model)
  - down (folds 1-10 with downsampling of B)
  - fic (folds 1-10 for coarse-grained model fictive-non-fictive)
  - predicted (predicted data test/large scale)
  - zeit (zeit text snippets for large scale)
  - dvjs (dvjs text snippets for large scale)
- scripts
  - text classification (doc_classification*)
  - inference (erzaehler_inference*)
  - extract_erzaehler (to extract text snippets from corpora)
  - get_erzaehler_embeddings (to plot token embeddings cf. Figure 4)
  - make_confusion_test (plot confusion matrix on predicted test data)

## Train your own models
Training models is based on deepset FARM: https://github.com/deepset-ai/FARM

Clone their repository and make sure that the correct dependencies are used (sentence-transformers has since gotten a dedicated version number):
```
git clone https://github.com/deepset-ai/FARM.git
cd FARM
pip install -r requirements.txt
pip install --editable .
```
Make sure to set the correct fold number in the script!
