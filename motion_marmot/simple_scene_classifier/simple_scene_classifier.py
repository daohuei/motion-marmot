import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


class SimpleSceneClassifier:
    """
    Simple Scene Classifier
    Description: Basically using K-Nearest Neighbors to train the classifier with the data 
    of several scene motion mask attributes
    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"SimpleSceneClassifier(name={self.name})"

    def data_extraction(self, file_name):
        """Extract Data from CSV File"""
        data = pd.read_csv(file_name)
        x = data.iloc[:, :-1].values
        y = data['scene']
        return x, y

    def train_model(self, train_x, train_y):
        return KNeighborsClassifier(n_neighbors=4).fit(train_x, train_y)

    def save_model(self, model, model_name):
        from joblib import dump
        dump(model, model_name)

    def load_model(self,  model_name):
        from joblib import load
        model = load(model_name)
        return model
