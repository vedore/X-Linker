import os
import pandas as pd

from sklearn.cluster import AgglomerativeClustering as SKAC
from sklearn.linear_model import LogisticRegression

from src.machine_learning.clustering import Clustering

class AgglomerativeClusteringCPU(Clustering):
    
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

    def save_labels(self, clustering_folder):
        os.makedirs(clustering_folder, exist_ok=True)
        clustering_df = pd.Dataframe(
            {
                'Labels': self.model.labels_.to_numpy()
            }
        )
        clustering_df.to_parquet(os.path.join(clustering_df, 'labels.parquet'))

    def get_labels(self):
        return self.model.labels_
        
class LogisticRegressionCPU(Clustering):

    @classmethod
    def train(cls, X_train, y_train, **kwargs):
        defaults = {
            'max_inter': 1000,
            'random_state': 42,
        }

        model = LogisticRegression(**defaults)
        model.fit(X_train, y_train)
        return cls(model=model, model_type='Logistic')