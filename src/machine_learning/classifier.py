import os
import pickle

class Classifier():

    def __init__(self, model=None, model_type=None):
        self.model = model
        self.model_type = model_type

    def save(self, classifier_folder):
        os.makedirs(classifier_folder, exist_ok=True)
        with open(os.path.join(classifier_folder, 'classifier.pkl'), 'wb') as fout:
            pickle.dump({'model': self.model, 'model_type': self.model_type}, fout)
    
    @classmethod
    def load(cls, classifier_folder):
        classifier_path = os.path.join(classifier_folder, 'classifier.pkl')
        assert os.path.exists(classifier_path), f"{classifier_path} does not exist"
        with open(classifier_path, 'rb') as fclu:
            data = pickle.load(fclu)
        return cls(model=data['model'], model_type=data['model_type'])    