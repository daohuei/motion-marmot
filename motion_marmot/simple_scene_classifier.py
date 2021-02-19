import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import typer


class SimpleSceneClassifier:
    """
    Simple Scene Classifier
    Description: Basically using K-Nearest Neighbors to train the classifier with the data 
    of several scene motion mask attributes
    """

    def __init__(self, name, model: str = None):
        self.name = name
        if model:
            self.model = self.load_model(model)

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

    def predict(self, avg, std, width, height):
        data = {
            'avg': avg,
            'std': std,
            'width': width,
            'height': height
        }
        record = pd.DataFrame.from_records([data])
        prediction = self.model.predict(record)
        return prediction[0]


app = typer.Typer()


@app.command()
def train(train_data: str):
    ssc = SimpleSceneClassifier('train')
    train_x, train_y = ssc.data_extraction(train_data)
    scene_knn_model = ssc.train_model(train_x, train_y)
    ssc.save_model(scene_knn_model, 'model/scene_knn_model')


@app.command()
def test(test_data: str):
    from sklearn.metrics import accuracy_score
    ssc = SimpleSceneClassifier('test', 'model/scene_knn_model')
    test_x, test_y = ssc.data_extraction(test_data)
    scene_knn_model = ssc.model
    knn_prediction = scene_knn_model.predict(test_x)
    accuracy = accuracy_score(knn_prediction, test_y)
    print(accuracy)


@app.command()
def test_record():
    ssc = SimpleSceneClassifier('test_record', 'model/scene_knn_model')
    scene_knn_model = ssc.model
    data = {
        'avg': 0.33678756476683935,
        'std': 2.415256783020859,
        'width': 1920,
        'height': 1080
    }
    record = pd.DataFrame.from_records([data])
    print(record)
    prediction = scene_knn_model.predict(record)
    print(prediction[0])


def main():
    """Main program"""
    app()


if __name__ == "__main__":
    main()
