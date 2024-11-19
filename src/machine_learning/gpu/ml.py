import os
import pickle
import subprocess


try:
    subprocess.check_output('nvidia-smi')
    GPU_AVAILABLE = True
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    print('No Nvidia GPU in system!')
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    import cupy as cp
    import cudf
    from cuml.cluster import AgglomerativeClustering as CUAC


class ClusteringGPU():

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

class AgglomerativeClustering(ClusteringGPU):

    @classmethod
    def train(cls, embeddings):
        defaults = {
            'n_clusters': 15,
            'metric': 'euclidean',
            'linkage': 'single',
        }

        # defaults.update(kwargs)

        model = CUAC(**defaults)
        embeddings = cudf.DataFrame(embeddings.toarray())
        model.fit(embeddings)

        return cls(model=model, model_type='Agglomerative')
    
    def get_labels(self):
        return self.model.labels_.to_numpy()