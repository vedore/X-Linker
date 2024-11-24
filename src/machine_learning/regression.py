import os
import pickle

class Regression():

    def __init__(self, model=None, model_type=None):
        self.model = model
        self.model_type = model_type

    def save(self, regression_folder):
        os.makedirs(regression_folder, exist_ok=True)
        with open(os.path.join(regression_folder, 'regression.pkl'), 'wb') as fout:
            pickle.dump({'model': self.model, 'model_type': self.model_type}, fout)
    
    @classmethod
    def load(cls, regression_folder):
        regression_path = os.path.join(regression_folder, 'clustering.pkl')
        assert os.path.exists(regression_path), f"{regression_path} does not exist"
        with open(regression_path, 'rb') as fclu:
            data = pickle.load(fclu)
        return cls(model=data['model'], model_type=data['model_type'])    