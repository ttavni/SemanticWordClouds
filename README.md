# Semantic Word Cloud

###### Made by Peter Simkin (Psimkin) & Tim Avni (tavni96)

Using keyword extraction (PKE) and pre-trained word embeddings to create a semantic word cloud

```python

from SemanticCloud.keywords import keywordExtractor
from SemanticCloud.wordembeddings import word2vec
from SemanticCloud.reduction import Mapper
from SemanticCloud.viz import createViz
import pandas as pd
import numpy as np

text = # Big old text document

# Get keywords, embeddings and map to 2D
keywords = keywordExtractor(text)
embeddings = word2vec(keywords)
data = Mapper(embeddings)

# Putting the data together and removing outliers
df = pd.merge(data, pd.DataFrame.from_dict(keywords, columns=['value'], orient='index'), left_index=True, right_index=True)
df = df[np.abs(df.X - df.X.mean()) <= (2 * df.X.std())]
df = df[np.abs(df.Y - df.Y.mean()) <= (2 * df.Y.std())]

# Create Visualisation
createViz(df,folder=folder)


```

[Interactive Version](http://bl.ocks.org/tavni96/raw/68d6ee63385d1a5a22a595efb3440378/6530e477ce5a979db1e32435b7ead1ffb5d66401/) (scroll out a little bit)


!['viz'](https://i.imgur.com/nr2pgc3.png)


To authenticate vecshare you will need to run this command in your terminal:
``` dw configure ``` and retrieve an API token from [data.world](https://data.world/)