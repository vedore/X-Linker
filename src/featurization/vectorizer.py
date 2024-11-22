import numpy as np
import torch
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVec
from transformers import AutoTokenizer, AutoModel

from src.featurization.preprocessor import Preprocessor


class TfidfVectorizer(Preprocessor):
    
    # Had an config file
    """
    Pecos config file for the tfidf in C 
        {"type": "tfidf", "kwargs": 
            {"base_vect_configs": 
                [{  "ngram_range": [1, 2], 
                    "truncate_length": -1, 
                    "max_feature": 0, 
                    "min_df_ratio": 0.0, 
                    "max_df_ratio": 0.98, 
                    "min_df_cnt": 0, 
                    "max_df_cnt": -1, 
                    "binary": false, 
                    "use_idf": true, 
                    "smooth_idf": true, 
                    "add_one_idf": false, 
                    "sublinear_tf": false, 
                    "keep_frequent_feature": true, 
                    "norm": "l2", 
                    "analyzer": "word", 
                    "buffer_size": 0, 
                    "threads": 30, 
                    "norm_p": 2, 
                    "tok_type": 10, 
                    "max_length": -1, 
                    "min_ngram": 1, 
                    "max_ngram": 2}
                ]}
            }
        }

    """
    @classmethod
    def train(cls, trn_corpus, dtype=np.float32):
        defaults = {
            'encoding': 'utf-8',
            'stip_accents': 'unicode',
            'stop_words': None, 
            'ngram_range': (1, 1),
            'min_df': 1,
            'lowercase': True,
            'norm': 'l2',
            'dype': dtype     
        }
        try:
            model = TfidfVec(**defaults)
        except TypeError:
            raise Exception(
                f"vectorizer config {defaults} contains unexpected keyword arguments for TfidfVectorizer"
            )
        model.fit(trn_corpus)
        return cls(vectorizer=model, vectorizer_type='tfidf')

class BioBertVectorizer():

    model_name = "dmis-lab/biobert-base-cased-v1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    @classmethod
    def predict(cls, corpus):
        inputs = cls.tokenizer(corpus, return_tensors='pt', padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = cls.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)




