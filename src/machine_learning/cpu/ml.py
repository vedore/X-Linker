import os
import pickle

from sklearn.cluster import AgglomerativeClustering as SKAC

class ClusteringCPU():

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
    
class AgglomerativeClustering(ClusteringCPU):
    
    @classmethod
    def train(cls, embeddings, **kwargs):
        defaults = {
            'n_clusters': None,           
            'distance_threshold': 0.5,  
            'metric': 'euclidean',
            'linkage': 'ward',
        }

        # defaults.update(kwargs)

        model = SKAC(**defaults)
        model.fit(embeddings.toarray())

        return cls(model=model, model_type='Agglomerative')
    
    def get_labels(self):
        return self.model.labels_
        
    