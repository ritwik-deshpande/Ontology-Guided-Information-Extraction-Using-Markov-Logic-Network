import keras
from Parsers import getDocuments
from Converters import getGoldTruthTest,getGoldTruth,getIndexes,getRelations
from KerasTextClassifier.models import KerasTextClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import csv
import sys

if __name__ == '__main__':
         
    documents_train = getDocuments('../Data/train.txt')
    relations_train = getRelations(documents_train)
    gold_truths_train,tr_tr,ts_tr = getGoldTruth(relations_train,documents_train)
    documents_test = getDocuments('../Data/test.txt')
    relations_test = getRelations(documents_test)
    gold_truths_test,tr_te,ts_te = getGoldTruth(relations_test,documents_test)
    n_relations = len(set(tr_tr))
	
    if sys.argv[1] == 'train':
        
        kclf = KerasTextClassifier(input_length=50, n_classes=n_relations, max_words=15000)
        tr_sent, te_sent, tr_rel, te_rel = train_test_split(ts_tr, tr_tr, test_size=0.1)
        kclf.fit(X=tr_sent, y=tr_rel, X_val=te_sent, y_val=te_rel,batch_size=64, lr=0.001, epochs=50)
        kclf.save()

    if sys.argv[1] == 'test':
  
        kclf = KerasTextClassifier(input_length=50, n_classes=n_relations, max_words=15000)
        kclf.load()
        label_to_use = list(kclf.encoder.classes_)
        pointer = 0;
        data = pd.read_csv('../Data/base_classifier_test.csv')
        
        actual_tags=[]
        predicted_tags=[]

        with open('../Data/relation_classifier_test.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["SentenceID","token1","token2","Sentence","Predicted","Gold Truth"])
            for sentenceID in range(0,288):
                input_sentences = []
                gold_truth_indexes =[]
                y_test_pred = []
                data_to_send = data.loc[data['SentenceID'] == sentenceID+1]
                input_sentences,gold_truth_indexes = getIndexes(data_to_send,"Gold Truth",pointer)
                pointer = pointer + len(data_to_send['Word'])
                if len(gold_truth_indexes) > 0:
                    y_test_pred = kclf.predict(input_sentences)
                    for i in range(0,len(y_test_pred)):
                        writer.writerow([sentenceID+1,gold_truth_indexes[i][0],gold_truth_indexes[i][1],input_sentences[i],label_to_use[y_test_pred[i]]
                                                                                                                     ,getGoldTruthTest(gold_truths_test,sentenceID,gold_truth_indexes[i])])
                        predicted_tags.append(label_to_use[y_test_pred[i]])
                        actual_tags.append(getGoldTruthTest(gold_truths_test,sentenceID,gold_truth_indexes[i]))

        print("relation_classifier_test.csv has been created")							
        print('classification_report: \n',classification_report(actual_tags,predicted_tags,digits=3))
        print('confusion_matrix :\n',confusion_matrix(actual_tags,predicted_tags))	
