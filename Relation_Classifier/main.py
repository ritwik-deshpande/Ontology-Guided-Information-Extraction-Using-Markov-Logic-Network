import keras
# from tensorflow.python.keras.models import KerasTextClassifier
from models import KerasTextClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import os
#print(os.getcwd())
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.metrics import f1_score, classification_report, accuracy_score
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
import csv
%matplotlib inline
from keras.preprocessing.sequence import pad_sequences


documents_test = getDocuments('test.txt')    
documents_train = getDocuments('train.txt')

relations_test = getRelations(documents_test)
relations_train = getRelations(documents_train)

gold_truths_test,tr_te,ts_te = getGoldTruth(relations_test,documents_test)
gold_truths_train,tr_tr,ts_tr = getGoldTruth(relations_train,documents_train)

n_relations = len(set(tr_tr))

kclf = KerasTextClassifier(input_length=50, n_classes=n_relations, max_words=15000)
tr_sent, te_sent, tr_rel, te_rel = train_test_split(ts_tr, tr_tr, test_size=0.1)
kclf.fit(X=tr_sent, y=tr_rel, X_val=te_sent, y_val=te_rel,
         batch_size=64, lr=0.001, epochs=50)

kclf.save_model()
kclf.save()

kclf = KerasTextClassifier(input_length=50, n_classes=n_relations, max_words=15000)
kclf.load()
tr_sent, te_sent, tr_rel, te_rel = train_test_split(ts_tr, tr_tr, test_size=0.1)

