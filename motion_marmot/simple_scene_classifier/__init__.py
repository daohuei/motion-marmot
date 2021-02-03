import numpy as np


class SimpleSceneClassifier:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"SimpleSceneClassifier(name={self.name})"

    def load_csv(self, file_name):
        return np.genfromtxt(file_name, delimiter=',')
