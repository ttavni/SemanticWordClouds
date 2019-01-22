from vecshare import vecshare as vs
from nltk import word_tokenize
import numpy as np
import pandas as pd

def word2vec(text_dict,embedding_model='text8_emb'):
	""" Extract word embeddings for keywords """

	embeddings = {}
	text_list = [*text_dict]
	for text in text_list:
		sentence_split = word_tokenize(text)
		embeddings[text] = vs.query(sentence_split, embedding_model).mean().values

		if embeddings[text].size == 0:
			embeddings[text] = np.zeros(50,)

	embedding_df  = pd.DataFrame.from_dict(embeddings).T
	embedding_df = embedding_df[(embedding_df.T != 0).any()]

	return embedding_df