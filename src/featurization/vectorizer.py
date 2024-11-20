import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVec

from src.featurization.preprocessor import Preprocessor


class TfidfVectorizer(Preprocessor):
    
    @classmethod
    # Had an config file
    def train(cls, trn_corpus, dtype=np.float32):
        defaults = {
            'encoding': 'utf-8',
            'strip_accents': 'unicode',
            'stop_words': None,
            'ngram_range': (1, 1),
            'min_df': 1,
            'lowercase': True,
            'norm': 'l2',
            'dtype': dtype,
        }
        try:
            model = TfidfVec(**defaults)
        except TypeError:
            raise Exception(
                f"vectorizer config {defaults} contains unexpected keyword arguments for TfidfVectorizer"
            )
        model.fit(trn_corpus)
        return cls(vectorizer=model, vectorizer_type='tfidf')

