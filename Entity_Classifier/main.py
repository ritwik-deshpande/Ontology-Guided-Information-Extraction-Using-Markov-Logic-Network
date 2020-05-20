from Parsers import parseDocuments
from Converters import Converters
from EntityClassifier import EntityClassifier
import numpy as np
import sys
import csv
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing




if __name__=='__main__':
	words = []
	tags = []

	train_documents = []
	test_documents = []

	SENTENCE_LENGTH = 100
	WORD_LENGTH =10



	train_documents,train_words,train_tags = parseDocuments("../Data/train.txt")
	
	test_documents,test_words ,test_tags = parseDocuments("../Data/test.txt")
	

	words = sorted(list(set(train_words+test_words)))
	tags = sorted(list(set(train_tags+test_tags)))
        
	
	if sys.argv[1] == 'train':


		entity_classifier = EntityClassifier(SENTENCE_LENGTH,WORD_LENGTH,words,tags,label = 'train')

		entity_classifier.trainModel(train_documents,batch_size=32,epochs = 43)
		entity_classifier.save()
	


	if sys.argv[1] == 'test':

		entity_classifier = EntityClassifier(label= 'test')
		entity_classifier.load()
		y_pred,y_te,input_words = entity_classifier.predict(test_documents)

		total_results = len(y_pred)
	
		predicted_tags = []
		actual_tags = []

		tag_labels = entity_classifier.getTagLabels()
		word_labels = entity_classifier.getWordLabels()

		with open('../Data/base_classifier_test.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(["SentenceID","tokenID","Word", "Predicted","Gold Truth"])
			for i in range(0,total_results):
				p = np.argmax(y_pred[i], axis=-1)
			
				token_count = 0
				
				for w, pred in zip(input_words[i], p):
					if w != 0:
						predicted_tags.append(tag_labels[pred])
						actual_tags.append(tag_labels[y_te[i][token_count]])
						writer.writerow([i+1,token_count+1,word_labels[w], tag_labels[pred],tag_labels[y_te[i][token_count]]])
						token_count = token_count + 1


		print("base_classifier_test.csv has been created")
		print('classification_report: \n',classification_report(actual_tags,predicted_tags,digits=3))
		print('confusion_matrix :\n',confusion_matrix(actual_tags,predicted_tags))
						
						

