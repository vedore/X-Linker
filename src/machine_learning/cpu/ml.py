from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LogisticRegression

from src.machine_learning.clustering import Clustering
from src.machine_learning.classifier import Classifier

class AgglomerativeClusteringCPU(Clustering):
    
    @classmethod
    def train(cls, embeddings):
        defaults = {
            'n_clusters': 16,           
        }

        # defaults.update(kwargs)
        model = AgglomerativeClustering(**defaults)
        model.fit(embeddings)
        return cls(model=model, model_type='HierarchicalCPU')

    def get_labels(self):
        return self.model.labels_
        
class LogisticRegressionCPU(Classifier):

    @classmethod
    def train(cls, X_train, y_train):
        defaults = {
            'random_state': 0,
            'solver': 'lbfgs',
            'max_iter': 100,
            'verbose': 0
        }

        # SVM
        model = LogisticRegression(**defaults)
        # X_train = csr_matrix(X_train)
        model.fit(X_train, y_train)
        return cls(model=model, model_type='LogisticRegressionCPU')
    
class KMeansCPU(Classifier):

    @classmethod
    def train(cls, X_train):
        defaults = {
            'n_clusters': 16,
            'max_iter': 300,
            'random_state': 0
        }
        # X_train -> Embeddings
        # If a sparse matrix is passed, a copy will be made if it’s not in CSR format.
        model = KMeans(**defaults)
        model.fit(X_train)
        return cls(model=model, model_type='KMeansCPU')
    
    def get_labels(self):
        return self.model.labels_