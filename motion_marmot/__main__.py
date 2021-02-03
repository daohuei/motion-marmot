from simple_scene_classifier import SimpleSceneClassifier
if __name__ == "__main__":
    print("=====Starting Motion Marmot=====")
    ssc = SimpleSceneClassifier("v1.0")
    train_data = ssc.load_csv("data/scene_train.csv")
    print(train_data)
    train_x = train_data[:, :-1]
    train_y = train_data[:, -1]
    print(train_x)
    print(train_y)
    print(train_x.shape)
    print(train_y.shape)
