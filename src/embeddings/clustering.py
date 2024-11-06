import os
import pandas as pd
import subprocess

try:
    subprocess.check_output('nvidia-smi')
    GPU_AVAILABLE = True
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    print('No Nvidia GPU in system!')
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    import cudf
    from cuml.cluster import AgglomerativeClustering as cuML_AgglomerativeClustering

from sklearn.cluster import AgglomerativeClustering as sk_AgglomerativeClustering

CLUSTERING_FOLDER = "data/processed/clustering"

class HierarchicalClustering():

    def __init__(self, kb_type, use_gpu=False):
        self.kb_type = kb_type
        self.use_gpu = use_gpu and GPU_AVAILABLE

        self.clustering_df = None

    def save(self, clustering_folder=CLUSTERING_FOLDER):
        if self.use_gpu:
            clustering_file = os.path.join(clustering_folder, f"{self.kb_type}_clustering_gpu.parquet")
        else:
            clustering_file = os.path.join(clustering_folder, f"{self.kb_type}_clustering_cpu.parquet")
        self.clustering_df.to_parquet(clustering_file)
    
    def hierarchical_clustering(self, embeddings, embeddings_ids):
        if self.use_gpu:
            clustering_model = cuML_AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                metric='cosine',
                linkage='average',
            )
            embeddings_df = cudf.Dataframe.from_records(embeddings)
            clustering_model.fit(embeddings_df)
        else:
            clustering_model = sk_AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                metric='cosine',
                linkage='average',
            )
            clustering_model.fit(embeddings)

        clustering_df = pd.DataFrame({
            'EntityID': embeddings_ids, 
            'ClusterLabel': clustering_model.labels_
        })

        self.clustering_df = clustering_df
        self.save()
        return clustering_df
