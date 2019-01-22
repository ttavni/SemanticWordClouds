# Semantic Word Cloud

Using keyword extraction (PKE) and pre-trained word embeddings to create a semantic word cloud

```python

from SemanticCloud.keywords import keywordExtractor
from SemanticCloud.wordembeddings import word2vec
from SemanticCloud.reduction import Mapper
from SemanticCloud.viz import createViz
import pandas as pd

text = # Big Document of Text (str)
keywords = keywordExtractor(text)
embeddings = word2vec(keywords)
data = Mapper(embeddings)
df = pd.merge(data, pd.DataFrame.from_dict(keywords, columns=['value'], orient='index'), left_index=True, right_index=True)
createViz(df,folder='test')

```
!['viz'](https://i.imgur.com/nr2pgc3.png)


To authenticate vecshare you will need to run this command in your terminal:
``` dw configure ``` and retrieve an API token from [data.world](https://data.world/)