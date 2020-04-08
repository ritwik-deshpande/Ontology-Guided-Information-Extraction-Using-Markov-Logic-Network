from keras.preprocessing.sequence import pad_sequences
import numpy as np

class Converters:
	def __init__(self,words,tags):
		self.word2idx = {w: i + 2 for i, w in enumerate(words)}
		self.word2idx["UNK"] = 1
		self.word2idx["PAD"] = 0
		self.tag2idx = {t: i + 1 for i, t in enumerate(tags)}
		self.tag2idx["PAD"] = 0
		self.idx2word = {i: w for w, i in self.word2idx.items()}
		self.idx2tag = {i: w for w, i in self.tag2idx.items()}


		
		self.char2idx = {c: i + 2 for i, c in enumerate(set([w_i for w in words for w_i in w]))}
		self.char2idx["UNK"] = 1
		self.char2idx["PAD"] = 0


	def getWord2Idx(self):
		return self.word2idx

	def getChar2Idx(self):
		return self.char2idx

	def getTag2Idx(self):
		return self.tag2idx

	def getIdx2Word(self):
		return self.idx2word

	def getIdx2Tag(self):
		return self.idx2tag


	def reformatDocuments(self,documents,doc_length,word_length):

		max_len = doc_length
		max_len_char = word_length

		X_word = [[self.word2idx[w[0]] for w in document] for document in documents]

		X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=self.word2idx["PAD"], padding='post', truncating='post')

		X_char = []
		for document in documents:
		    sent_seq = []
		    for i in range(max_len):
		        word_seq = []
		        for j in range(max_len_char):
		            try:
		                word_seq.append(self.char2idx.get(document[i][0][j]))
		            except:
		                word_seq.append(self.char2idx.get("PAD"))
		        sent_seq.append(word_seq)
		    X_char.append(np.array(sent_seq))


		y = [[self.tag2idx[word_label[1]] for word_label in document] for document in documents]
		y = pad_sequences(maxlen=max_len, sequences=y, value=self.tag2idx["PAD"], padding='post', truncating='post')

		return X_word, X_char, y