import typer
from simple_scene_classifier import SimpleSceneClassifier

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
    ssc = SimpleSceneClassifier('test')
    test_x, test_y = ssc.data_extraction(test_data)
    scene_knn_model = ssc.load_model('model/scene_knn_model')
    knn_prediction = scene_knn_model.predict(test_x)
    accuracy = accuracy_score(knn_prediction, test_y)
    print(accuracy)


def main():
    """Main program"""
    app()


if __name__ == "__main__":
    main()
