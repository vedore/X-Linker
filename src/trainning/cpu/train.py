from sklearn.model_selection import train_test_split

from src.machine_learning.cpu.ml import LogisticRegressionCPU
from src.trainning.metrics import Metrics

        
class TrainCPU():

    @classmethod
    def train(cls, embeddings, clustering_labels):
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, 
            clustering_labels['Labels'], 
            test_size=0.2, 
            random_state=42
            )
        
        y_train = y_train.to_numpy()
        model = LogisticRegressionCPU.train(X_train, y_train).model
        # cls.save(model, "data/processed/regression")

        Metrics.evaluate(model, X_train, y_train, X_test, y_test)

    """
    def save(model, regression_folder):
        os.makedirs(regression_folder, exist_ok=True)
        with open(os.path.join(regression_folder, 'regression.pkl'), 'wb') as fout:
            pickle.dump({'model': model, 'model_type': 'regression'}, fout)    

    @classmethod
    def load(cls, clustering_folder):
        clustering_path = os.path.join(clustering_folder, 'clustering.pkl')
        assert os.path.exists(clustering_path), f"{clustering_path} does not exist"
        with open(clustering_path, 'rb') as fclu:
            data = pickle.load(fclu)
        return cls(model=data['model'], model_type=data['model_type'])    
    """


