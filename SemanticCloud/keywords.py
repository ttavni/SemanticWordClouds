from pke.unsupervised import TopicRank

def keywordExtractor(text):
	""" Keyword extraction from text string """

	extractor = TopicRank()
	extractor.load_document(input=text, language="en", normalization=None)
	extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})
	extractor.candidate_weighting(threshold=0.5,method='average')
	keyword_counts_list = dict(extractor.get_n_best(n=len(extractor.candidates), stemming=True))
	return keyword_counts_list