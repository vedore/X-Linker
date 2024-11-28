import json
import pickle
import os

kb_dict = {
    'medic': 'DiseaseID',
    'chemical': 'ChemicalID'
    }

class Preprocessor():

    def __init__(self, model=None, model_type=None):
        self.model = model
        self.model_type = model

    def save(self, preprocessor_folder):
        os.makedirs(preprocessor_folder, exist_ok=True)
        with open(os.path.join(preprocessor_folder, 'vectorizer.pkl'), 'wb') as fout:
            pickle.dump({'model': self.model, 'model_type': self.model_type}, fout)

    @classmethod
    def load(cls, preprocessor_folder):
        preprocessor_path = os.path.join(preprocessor_folder, 'vectorizer.pkl')
        assert os.path.exists(preprocessor_path), f"{preprocessor_path} does not exist"
        with open(preprocessor_path, 'rb') as fclu:
            data = pickle.load(fclu)
        return cls(model=data['model'], model_type=data['model_type'])
    
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
    
    @staticmethod
    def load_labels_from_dict(labels_data):
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
        return self.model.transform(corpus)
    
        