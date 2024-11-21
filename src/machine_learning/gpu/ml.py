import subprocess
import os
import pandas as pd

from src.machine_learning.clustering import Clustering


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


class AgglomerativeClusteringGPU(Clustering):

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

    def save_labels(self, clustering_folder):
        os.makedirs(clustering_folder, exist_ok=True)
        clustering_df = pd.DataFrame(
            {
                'Labels': self.model.labels_.to_numpy()
            }
        )
        clustering_df.to_parquet(os.path.join(clustering_folder, 'labels.parquet'))
    
    def get_labels(self):
        return self.model.labels_.to_numpy()