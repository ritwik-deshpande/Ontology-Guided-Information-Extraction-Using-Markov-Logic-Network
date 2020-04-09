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

if __name__ == '__main__':
         
         documents_train = getDocuments('train.txt')
         relations_train = getRelations(documents_train)
         gold_truths_train,tr_tr,ts_tr = getGoldTruth(relations_train,documents_train)
	n_relations = len(set(tr_tr))
         
         if sys.argv[1] == 'train':
                  kclf = KerasTextClassifier(input_length=50, n_classes=n_relations, max_words=15000)
                  tr_sent, te_sent, tr_rel, te_rel = train_test_split(ts_tr, tr_tr, test_size=0.1)
                  kclf.fit(X=tr_sent, y=tr_rel, X_val=te_sent, y_val=te_rel,batch_size=64, lr=0.001, epochs=50)
                  kclf.save_model() 
                  kclf.save()


	if sys.argv[1] == 'test':
      
		kclf = KerasTextClassifier(input_length=50, n_classes=n_relations, max_words=15000)
                  kclf.load()
                  label_to_use = list(kclf.encoder.classes_)
                  pointer = 0;
                  data = pd.read_csv('base_classifier_test.csv')
                  print(label_to_use)
                  with open('relation_classifier_test.csv', 'w', newline='') as file:
                           writer = csv.writer(file)
                           writer.writerow(["SentenceID","token1","token2","Sentence","Predicted","Gold Truth"])
                           for sentenceID in range(0,288):
                                    input_sentences = []
                                    gold_truth_indexes =[]
                                    y_test_pred = []
                                    data_to_send = data.loc[data['SentenceID'] == sentenceID+1]
                                    print(data_to_send)
                                    input_sentences,gold_truth_indexes = getIndexes(data_to_send,"Gold Truth",pointer)
                                    pointer = pointer + len(data_to_send['Word'])
                                    print(input_sentences)
                                    if len(gold_truth_indexes) > 0:
                                             y_test_pred = kclf.predict(input_sentences)
                                             for i in range(0,len(y_test_pred)):
                                                      writer.writerow([sentenceID+1,gold_truth_indexes[i][0],
                                                                       gold_truth_indexes[i][1],
                                                      input_sentences[i],label_to_use[y_test_pred[i]],
                                                                       getGoldTruthTest(sentenceID,gold_truth_indexes[i])])
