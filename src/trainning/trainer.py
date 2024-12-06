from src.machine_learning.classifier import Classifier


class Trainer():

    # Will receive an Classifier
    def __init__(self, classifier:Classifier):
        self.model = classifier.model
        self.model_name = classifier.model_type

        