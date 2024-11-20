import os
import pickle
import pandas as pd

class Clustering():

    def __init__(self, model=None, model_type=None):
        self.model = model
        self.model_type = model_type

    def save(self, clustering_folder):
        os.makedirs(clustering_folder, exist_ok=True)
        with open(os.path.join(clustering_folder, 'clustering.pkl'), 'wb') as fout:
            pickle.dump({'model': self.model, 'model_type': self.model_type}, fout)
    
    @classmethod
    def load(cls, clustering_folder):
        clustering_path = os.path.join(clustering_folder, 'clustering.pkl')
        assert os.path.exists(clustering_path), f"{clustering_path} does not exist"
        with open(clustering_path, 'rb') as fclu:
            data = pickle.load(fclu)
        return cls(model=data['model'], model_type=data['model_type'])    
    
    @staticmethod
    def load_labels(clustering_folder):
        clustering_path = os.path.join(clustering_folder, 'labels.parquet')
        assert os.path.exists(clustering_path), f"{clustering_path} does not exist"
        return pd.read_parquet(clustering_path)