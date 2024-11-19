import json
import numpy as np
import pandas as pd
import os

from src.featurization.vectorizer import Vectorizer

kb_dict = {
    'medic': 'DiseaseID',
    'chemical': 'ChemicalID'
    }

class Preprocessor(object):

    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer

    def save(self, preprocessor_folder):
        self.vectorizer.save(preprocessor_folder)

    @classmethod
    def load(cls, preprocessor_folder):
        vectorizer = Vectorizer.load(preprocessor_folder)
        return cls(vectorizer)
    
    @classmethod
    ## Had an config file
    def train(cls, corpus, dtype=np.float32):
        vectorizer = Vectorizer.train(corpus, dtype=dtype)
        return cls(vectorizer)
    
    def predict(self, corpus, **kwargs):
        return self.vectorizer.predict(corpus, **kwargs)
    
    @staticmethod
    def load_data_from_file(labels_folder):

        labels_file = os.path.join(labels_folder, 'labels.json')

        with open(labels_file, 'r') as json_file:
            labels_data = json.load(json_file)

        labels_dict = {}
        for labels_id, entries in labels_data.items():
            # Combine names and synonyms into a single text string for each entity
            names_and_synonyms = [entry.split(': ', 1)[1] for entry in entries]
            combined_text = ' '.join(names_and_synonyms)
            labels_dict[labels_id] = combined_text
        
        processed_labels_data = list(labels_dict.values())
        processed_labels_id = list(labels_dict.keys())

        return (processed_labels_id, processed_labels_data)
        