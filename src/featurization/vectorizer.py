import os
import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer():
    
    def __init__(self, model=None):
        self.model = model

    def save(self, vectorizer_folder):
        os.makedirs(vectorizer_folder, exist_ok=True)
        with open(os.path.join(vectorizer_folder, 'vectorizer.pkl'), 'wb') as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, vectorizer_folder):
        vectorizer_path = os.path.join(vectorizer_folder, 'vectorizer.pkl')
        assert os.path.exists(vectorizer_path), f"{vectorizer_path} does not exist"
        with open(vectorizer_path, 'rb') as fvec:
            return cls(pickle.load(fvec))
    
    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
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
            model = TfidfVectorizer(**{**defaults, **config})
        except TypeError:
            raise Exception(
                f"vectorizer config {config} contains unexpected keyword arguments for TfidfVectorizer"
            )
        model.fit(trn_corpus)
        return cls(model)
    
    def predict(self, corpus):
        result = self.model.transform(corpus)
        return result

