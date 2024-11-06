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
    import cupy as cp
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
            # Initialize cuML clustering model
            clustering_model = cuML_AgglomerativeClustering(
                n_clusters=2,
                metric='cosine',
                linkage='single'
            )

            # Convert embeddings to a cupy array if they are sparse
            embeddings = cp.asarray(embeddings.toarray())  # assuming embeddings is a sparse matrix
            embeddings_df = cudf.DataFrame.from_records(embeddings)  # Convert to cudf DataFrame
            
            # Fit model
            clustering_model.fit(embeddings_df)
        else:
            # Initialize scikit-learn clustering model for CPU
            clustering_model = sk_AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                metric='cosine',
                linkage='average',
            )
            
            # Fit model on CPU
            clustering_model.fit(embeddings)

         # Convert the cluster labels to a CPU-compatible format (NumPy array)
        labels = clustering_model.labels_.to_numpy() if self.use_gpu else clustering_model.labels_

        # Create DataFrame with EntityID and ClusterLabel
        clustering_df = pd.DataFrame({
            'EntityID': embeddings_ids, 
            'ClusterLabel': labels
        })

        # Store the results
        self.clustering_df = clustering_df
        print(self.clustering_df['ClusterLabel'])
        self.save()
        return clustering_df
