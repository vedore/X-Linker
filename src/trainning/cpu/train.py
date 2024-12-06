from sklearn.model_selection import train_test_split

from src.machine_learning.cpu.ml import LogisticRegressionCPU
from src.trainning.metrics import Metrics

        
class TrainCPU():

    def train(embeddings, clustering_labels):
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, 
            clustering_labels['Labels'], 
            test_size=0.2, 
            random_state=42
            )
        
        y_train = y_train.to_numpy()
        classifier = LogisticRegressionCPU.train(X_train, y_train)
        classifier.save("data/processed/regression")

        Metrics.evaluate(classifier.model, X_train, y_train, X_test, y_test)

    def train_top_k(n_clusters, top_k):
        pass




