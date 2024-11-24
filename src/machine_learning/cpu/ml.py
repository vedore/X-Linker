import os
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression

from src.machine_learning.clustering import Clustering
from src.machine_learning.regression import Regression

class AgglomerativeClusteringCPU(Clustering):
    
    @classmethod
    def train(cls, embeddings):
        defaults = {
            'n_clusters': 16,           
        }

        # defaults.update(kwargs)
        model = AgglomerativeClustering(**defaults)
        model.fit(embeddings)
        return cls(model=model, model_type='Agglomerative')

    def get_labels(self):
        return self.model.labels_
        
class LogisticRegressionCPU(Regression):

    @classmethod
    def train(cls, X_train, y_train):
        defaults = {
            'random_state': 0,
            'solver': 'lbfgs',
            'max_iter': 100,
            'verbose': 1
        }

        # SVM
        model = LogisticRegression(**defaults)
        model.fit(X_train, y_train)
        return cls(model=model, model_type='LogisticRegressionCPU')