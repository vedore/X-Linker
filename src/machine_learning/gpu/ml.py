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
    import cudf
    import cupy as cp
    from cuml.cluster import AgglomerativeClustering
    from cuml.linear_model import LogisticRegression


class AgglomerativeClusteringGPU(Clustering):

    @classmethod
    def train(cls, embeddings):
        defaults = {
            'n_clusters': 16,
            'metric': 'euclidean',
        }

        # defaults.update(kwargs)
        model = AgglomerativeClustering(**defaults)
        embeddings = cudf.DataFrame(embeddings)
        model.fit(embeddings)
        return cls(model=model, model_type='HierarchicalGPU')
    
    def get_labels(self):
        return self.model.labels_.to_numpy()
    
class LogisticRegressionGPU(Clustering):

    @classmethod
    def train(cls, x_train, y_train, **kwargs):
        defaults = {}
        model = LogisticRegression(**defaults)

        subset_size = 5000

        cp.get_default_memory_pool().free_all_blocks()
        model.fit(x_train[:subset_size], y_train[:subset_size])
        cp.get_default_memory_pool().free_all_blocks()
        return cls(model=model, model_type='LogisticRegressionGPU')