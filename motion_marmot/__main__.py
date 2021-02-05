from simple_scene_classifier import SimpleSceneClassifier
import numpy as np
if __name__ == "__main__":
    print("=====Starting Motion Marmot=====")
    ssc = SimpleSceneClassifier("v1.0")
    train_data = ssc.load_csv("data/scene_train.csv")
    test_data = ssc.load_csv("data/scene_test.csv")
    train_x = train_data[:, :-1]
    train_y = train_data[::, -1:]
    test_x = test_data[:, :-1]
    test_y = test_data[::, -1:]
    train_pos_bool = train_y == 3
    train_neg_bool = train_y != 3
    train_y[train_pos_bool] = 1
    train_y[train_neg_bool] = -1
    test_pos_bool = test_y == 3
    test_neg_bool = test_y != 3
    test_y[test_pos_bool] = 1
    test_y[test_neg_bool] = -1

    # add "intercept term:"
    train_x = np.hstack((np.ones((train_x.shape[0], 1)), train_x))
    test_x = np.hstack((np.ones((test_x.shape[0], 1)), test_x))

    # cross validation
    lam_val = np.logspace(2, 3, 100)
    res = list()
    for lam in lam_val:
        valid_result = ssc.cross_valid(
            train_x, train_y,
            lambda tr_x, tr_y, te_x, te_y:
                ssc.test_lr(te_x, te_y, ssc.learn_lr(tr_x, tr_y, lam))
        )
        res.append(valid_result)

    # Get the lambda value with the minimal loss
    best_lambda = lam_val[res.index(min(res))]
    print(f"The best learning rate is: {best_lambda}")

    # train and test
    classifier = ssc.learn_lr(train_x, train_y, best_lambda)
    test_results = ssc.test_lr(test_x, test_y, classifier)
    print(f"The Accuracy of Simple Scene Classifier: {(1-test_results)*100} %")
