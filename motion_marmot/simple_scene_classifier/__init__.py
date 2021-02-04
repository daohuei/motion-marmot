import numpy as np


class SimpleSceneClassifier:
    """
    Simple Scene Classifier
    Description: Basically using Logistic Regression to train the classifier with the data 
    of several scene motion mask attributes
    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"SimpleSceneClassifier(name={self.name})"

    def load_csv(self, file_name):
        return np.genfromtxt(file_name, delimiter=',')

# ===============
# Below codes were inspired by the professor Christian Shelton in the class of
# CS229: Machine Learning in UC Riverside
# ===============

    def graddesc(self, weight, fn, grad_fn, eta=0.1, ittfn=None):
        """Gradient Descent

        The function will help reach the target of the minimal loss.

        Args:
            weight (np.array): the start weight.
            fn (func): old loss function.
            grad_fn (func): the gradient function(the differentiation of the loss function).
            eta (float): learning rate, starting value for eta (no need to adjust).
            ittfn (func): ittfn is optional function to call on each iteration.
                        (no need to use, except if you wish for debugging)
        Returns:
            weight (np.array): return the best weight with minimal loss
        """
        x = weight  # weight
        oldf = fn(x)  # loss function
        df = 1  # derivative of function
        mineta = 1e-16  # minimum eta
        while(df > 1e-6):  # while derivative is large than a threshold
            g = grad_fn(x)  # get gradient of x
            while eta > mineta:  # while eta larger than a threshold(min eta)
                newx = x - eta*g  # calculate new x for input of the function
                newf = fn(newx)  # calculate new function
                # if old function(loss) > new function * 0.001 + a threshold
                if oldf > newf+abs(newf)*0.001+1e-6:
                    break  # then break to calculate another derivative
                eta *= 0.5  # decrease learning rate
            if ittfn is not None:  # displayprogress
                ittfn(x, eta, newf)
            if eta <= mineta:  # if less than mineta
                break  # break
            oldf = newf  # next round
            x = newx  # next round
        return x

    def mysigmoidf(self, X):
        """The basic function of Logistic Regression"""
        return lambda w: self.sigmoid(X@w)

    def sigmoid(self, f):
        """The Sigmoid Function"""
        return 1/(1+np.exp(-1*f))

    def loss_log_likelihood(self, X, Y, lam):
        """
        We use log likelihood as loss function of Logistic Regression
        (Only suitable in binary classification)
        """
        return lambda w: np.sum(-1*(np.log(self.sigmoid(Y*(X@w))))) + (lam/2) * (w.T@w)

    def grad_log_likelihood(self, X, Y, lam):
        """
        The gradient function of log likelihood
        """
        return lambda w: (-1 * (((1-self.sigmoid(Y*(X@w))).T@(X*Y))).T) + lam * w
    # X is m-by-n, Y is m-by-1, lam is a scalar (lambda of a regularization term lambda/2 * w^T * w)
    # returns a n-by-1 vector of weights

    def learn_lr(self, X, Y, lam):
        """Training function"""
        I = np.eye(X.shape[1])
        w0 = np.linalg.inv(X.T@X + I*lam)@(X.T@Y)  # start point of weight
        wmin = self.graddesc(w0, self.loss_log_likelihood(X, Y, lam),
                             self.grad_log_likelihood(X, Y, lam), 0.1)
        return wmin

    # X is m-by-n, Y is m-by-1, w is n-by-1
    # returns the average 0-1 error on the input data set.  That is, it returns the fraction of examples misclassified
    def test_lr(self, X, Y, w):
        """Testing function"""
        f = Y*(X@w)
        pos_bool = f > 0
        neg_bool = f <= 0

        res = f.copy()
        res[pos_bool] = 0
        res[neg_bool] = 1
        return np.sum(res)/X.shape[0]
