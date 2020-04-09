from tensorflow.python.keras.models import Model, Input
from tensorflow.python.keras.layers import LSTM, Embedding, Dense, TimeDistributed,concatenate,SpatialDropout1D,Bidirectional
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class EntityClassifier:


	def __init__(self,nwords = None,nchars=None,ntags=None,sentence_length=None,word_length=None,embeddings=None,label=None):
		if label == 'train':
			self.word_length = word_length
			self.sentence_length = sentence_length
			self.model = self.getModel(nwords,nchars,ntags,sentence_length,word_length,embeddings)
		else:
			print('Loading Pre-Trained Model for testing......')


	def getModel(self,nwords,nchars,ntags,max_len,max_len_char,embedding_matrix):
		word_in = Input(shape=(max_len,))


		emb_word = Embedding(nwords + 1 ,len(embedding_matrix[0]),
		                            weights=[embedding_matrix],
		                            input_length=max_len,
		                            trainable=False)(word_in)




		# input and embeddings for characters
		char_in = Input(shape=(max_len, max_len_char,))
		emb_char = TimeDistributed(Embedding(input_dim=nchars + 2, output_dim=10, input_length=max_len_char, mask_zero=True))(char_in)


		# character LSTM to get word encodings by characters
		char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,recurrent_dropout=0.68))(emb_char)

		x = concatenate([emb_word, char_enc])
		x = SpatialDropout1D(0.3)(x)
		main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,recurrent_dropout=0.68))(x)
		out = TimeDistributed(Dense(ntags + 1, activation="softmax"))(main_lstm)

		model = Model([word_in, char_in], out)
		model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

		model.summary()

		return model

	def save_model(self,model_file):
		self.model.save(model_file)

	def load_model(self,model_file):
		self.model = tf.keras.models.load_model(model_file)

	

	def trainModel(self,train_words,train_chars,y_labels,batch_size,epochs):
		history = self.model.fit([train_words,
                     np.array(train_chars).reshape((len(train_chars), self.sentence_length, self.word_length))],
                    np.array(y_labels).reshape(len(y_labels), self.sentence_length, 1),
                    batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)


		

		hist = pd.DataFrame(history.history)
		plt.style.use("ggplot")
		plt.figure(figsize=(12,12))
		plt.plot(hist["acc"])
		# plt.plot(hist["val_acc"])
		plt.show()

	def predict(self,input_word_documents,input_character_documents):

		 return self.model.predict([np.array(input_word_documents), np.array(input_character_documents)])

