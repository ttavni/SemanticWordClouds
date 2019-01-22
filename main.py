from SemanticCloud.keywords import keywordExtractor
from SemanticCloud.wordembeddings import word2vec
from SemanticCloud.reduction import Mapper
from SemanticCloud.viz import createViz
import pandas as pd
import numpy as np

folder = 'trump'
with open('data/trump_hillary_debate.txt', 'r') as f:
	text = f.read()

keywords = keywordExtractor(text)
embeddings = word2vec(keywords)
data = Mapper(embeddings)
df = pd.merge(data, pd.DataFrame.from_dict(keywords, columns=['value'], orient='index'), left_index=True, right_index=True)
df = df[np.abs(df.X - df.X.mean()) <= (2 * df.X.std())]
df = df[np.abs(df.Y - df.Y.mean()) <= (2 * df.Y.std())]
createViz(df,folder=folder)

