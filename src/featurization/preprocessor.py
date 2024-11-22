import json
import numpy as np
import pickle
import os

kb_dict = {
    'medic': 'DiseaseID',
    'chemical': 'ChemicalID'
    }

class Preprocessor():

    def __init__(self, vectorizer=None, vectorizer_type=None):
        self.vectorizer = vectorizer
        self.vectorizer_type = vectorizer_type

    def save(self, preprocessor_folder):
        os.makedirs(preprocessor_folder, exist_ok=True)
        with open(os.path.join(preprocessor_folder, 'vectorizer.pkl'), 'wb') as fout:
            pickle.dump({'vectorizer': self.vectorizer, 'vectorizer_type': self.vectorizer_type}, fout)

    @classmethod
    def load(cls, preprocessor_folder):
        preprocessor_path = os.path.join(preprocessor_folder, 'vectorizer.pkl')
        assert os.path.exists(preprocessor_path), f"{preprocessor_path} does not exist"
        with open(preprocessor_path, 'rb') as fclu:
            data = pickle.load(fclu)
        return cls(vectorizer=data['vectorizer'],vectorizer_type=data['vectorizer_type'])
    
    @staticmethod
    def load_labels_from_file(labels_folder):
        labels_file = os.path.join(labels_folder, 'labels.json')

        with open(labels_file, 'r') as json_file:
            labels_data = json.load(json_file)

        labels_dict = {}
        for labels_id, entries in labels_data.items():
            names_and_synonyms = " ".join(entries)
            unique_words = list(set(names_and_synonyms.split()))
            combined_text = ' '.join(unique_words)
            labels_dict[labels_id] = combined_text
        
        processed_labels_data = list(labels_dict.values())
        processed_labels_id = list(labels_dict.keys())

        return (processed_labels_id, processed_labels_data)
    
    def predict(self, corpus):
        return self.vectorizer.transform(corpus)
    
        