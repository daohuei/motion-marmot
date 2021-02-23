import typer
import pandas as pd
from motion_marmot.simple_scene_classifier import SimpleSceneClassifier

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
