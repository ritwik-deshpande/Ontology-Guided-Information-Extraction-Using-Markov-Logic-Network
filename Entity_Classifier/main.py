from Parsers import parseDocuments
from Converters import Converters
from EntityClassifier import EntityClassifier
import numpy as np
import sys
import csv
from sklearn.metrics import confusion_matrix, classification_report

def getEmbeddingMatrix(word2idx):
	EMBEDDING_DIM = 50
	embedding_matrix = np.zeros((len(word2idx.keys()) +1, EMBEDDING_DIM))

	f = open('../Data/vecs.lc.over100freq.txt',encoding='utf8',errors='ignore')
	for line in f:
	    values = line.split(' ')
	    word = values[0]
	    if word.lower() in word2idx.keys():
	        embedding_matrix[word2idx[word]] = [float(value) for value in values[1:-1]]


	return embedding_matrix

if __name__=='__main__':
	words = []
	tags = []

	train_documents = []
	test_documents = []

	SENTENCE_LENGTH = 50
	WORD_LENGTH =10



	train_documents,train_words,train_tags = parseDocuments("../Data/train.txt")
	
	test_documents,test_words ,test_tags = parseDocuments("../Data/test.txt")
	

	words = list(set(train_words+test_words))
	tags = list(set(train_tags+test_tags))
        
	converters = Converters(words,tags)


	X_word_tr, X_char_tr, y_tr = converters.reformatDocuments(train_documents,SENTENCE_LENGTH,WORD_LENGTH)
	X_word_te, X_char_te, y_te = converters.reformatDocuments(test_documents,SENTENCE_LENGTH,WORD_LENGTH)


	NUMBER_OF_WORDS = len(converters.getWord2Idx().keys())
	NUMBER_OF_TAGS = len(converters.getTag2Idx().keys())
	NUMBER_OF_CHARS = len(converters.getChar2Idx().keys())
	embedding_matrix = getEmbeddingMatrix(converters.getWord2Idx())
	

	entity_classifier = EntityClassifier(NUMBER_OF_WORDS,NUMBER_OF_CHARS,NUMBER_OF_TAGS,SENTENCE_LENGTH,WORD_LENGTH,embedding_matrix)

	if sys.argv[1] == 'train':
		entity_classifier.trainModel(X_word_tr,X_char_tr,y_tr,32,1)
		entity_classifier.save_model('EntityClassifier2.h5')
	


	if sys.argv[1] == 'test':
		entity_classifier.load_model('EntityClassifierModel.h5')
		y_pred = entity_classifier.predict(X_word_te[:2],X_char_te[:2])

		total_results = len(y_pred)
	
		predicted_tags = []
		actual_tags = []


		with open('../Data/base_classifier_test.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(["SentenceID","tokenID","Word", "Predicted","Gold Truth"])
			for i in range(0,total_results):
				p = np.argmax(y_pred[i], axis=-1)
			
				token_count = 0
				
				for w, pred in zip(X_word_te[i], p):
					if w != 0:
						predicted_tags.append(converters.getIdx2Tag()[pred])
						actual_tags.append(converters.getIdx2Tag()[y_te[i][token_count]])
						writer.writerow([i+1,token_count+1,converters.getIdx2Word()[w], converters.getIdx2Tag()[pred],converters.getIdx2Tag()[y_te[i][token_count]]])
						token_count = token_count + 1


		print('classification_report: \n',classification_report(actual_tags,predicted_tags,digits=3))
		print('confusion_matrix :\n',confusion_matrix(actual_tags,predicted_tags))
						
						

