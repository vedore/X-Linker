import numpy as np
import pandas as pd

from src.featurization.vectorizer import Vectorizer

kb_dict = {
    'medic': 'DiseaseID'
    }

class Preprocessor(object):

    def __init__(self, kb_type, vectorizer=None):
        self.kb_type = kb_type
        self.vectorizer = vectorizer

    def save(self, preprocessor_folder):
        self.vectorizer.save(preprocessor_folder)

    @classmethod
    def load(cls, preprocessor_folder):
        vectorizer = Vectorizer.load(preprocessor_folder)
        return cls(vectorizer)
    
    @classmethod
    def train(cls, corpus, vectorizer_config=None, dtype=np.float32):
        vectorizer = Vectorizer.train(corpus, vectorizer_config, dtype=dtype)
        return cls(vectorizer)
    
    def predict(self, corpus, **kwargs):
        return self.vectorizer.predict(corpus, **kwargs)
    
    @staticmethod
    def load_data_from_file(self, kb_filepath):
        defaults = {
            'sep': '\t',
            'header': None,
            'names': self,
            'skiprows': 29,
            'names': get_column_names_from_tsv(kb_filepath, 29)
        }

        def get_column_names_from_tsv(filepath, skip_rows):
            with open(filepath, 'r') as fin:
                for _ in range(skip_rows - 1):
                    line = fin.readline()
                col_names = [str(item).strip() for item in line.split("\t")]
                col_names[0] = col_names[0].replace('#', '').strip()
            return col_names
        