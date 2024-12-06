import json
import pickle
import os
import pandas as pd

kb_dict = {
    'medic': 'DiseaseID',
    'chemical': 'ChemicalID'
    }

class Preprocessor():

    def __init__(self, model=None, model_type=None):
        self.model = model
        self.model_type = model_type

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
    
    @staticmethod 
    def load_data_from_file(train_filepath, labels_filepath):
        assert os.path.exists(train_filepath), f"{train_filepath} does not exist"
        assert os.path.exists(labels_filepath), f"{labels_filepath} does not exist"

        train_df = pd.read_csv(train_filepath, header=None, names=['id', 'train_name'], delimiter="\t")
        # Corpus
        train_data = train_df['train_name'].to_list()

        labels_df = pd.read_csv(labels_filepath, header=None, names=['label'])
        # Dense Matrix
        labels_data = labels_df['label'].to_list()

        return {
                'labels_data':labels_data, 
                'corpus': train_data
                }
    
        