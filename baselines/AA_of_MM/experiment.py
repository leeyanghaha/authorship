from models.svm_classifier import Svm

class Experiment:
    def __init__(self, epochs):
        self.epochs = epochs

    def run_experiment(self, **params):
        pass

    def valid(self):
        pass
