import numpy as np
from numpy import arange, ones, zeros, e
from numpy import where
from scipy import optimize


def generate_features(X, n):
    """ Generates all quadratic feature vectors. """

    new_data = X[:, :]

    for i in xrange(n):
        for j in xrange(i + 1):
            new_feature = X[:, i:i + 1] * X[:, j:j + 1]
            new_data = np.append(new_data,
                                 new_feature,
                                 axis=1)

    return new_data, new_data.shape[1]


def sigmoid(X):
    return 1.0 / (1.0 + e ** (-1.0 * X))


def cost(X, y, m, n, theta, reg_factor):
    """ Calculates the cost for a given theta vector """

    theta = np.reshape(theta, (n, 1))

    predictions = X.dot(theta)
    predictions[where(predictions >= 20)] = 20.0
    predictions[where(predictions <= -500)] = -500.0

    hypotheses = sigmoid(predictions)
    hypotheses[where(hypotheses == 1.0)] = 0.99999

    costvalue = (-1.0 / m) * np.sum(y * np.log(hypotheses) +
                                 (1.0 - y) * np.log(1 - hypotheses))

    regularization_term = reg_factor / (2.0 * m) * \
                            np.sum(theta[1:, :] * theta[1:, :])

    costvalue = costvalue + regularization_term

    return costvalue


def gradient(X, y, m, n, theta, reg_factor):
    """ Calculates the gradient of the cost function
        for the given theta vector. """

    theta = np.reshape(theta, (n, 1))

    predictions = X.dot(theta)
    predictions[where(predictions >= 20)] = 20.0
    predictions[where(predictions <= -500)] = -500.0

    hypotheses = sigmoid(predictions)
    hypotheses.shape = (m, 1)
    hypotheses[where(hypotheses == 1.0)] = 0.9999

    delta = hypotheses - y
    gradient = (1.0 / m) * X.T.dot(delta)

    regularization_terms = (reg_factor / m) * np.concatenate(
                                    (zeros((1, 1)), theta[1:, :]))
    gradient += regularization_terms

    return gradient.flatten()


def logistic_regression(X, y, m, n, reg_factor):
    """ Builds a logistic regression model by minimizing
        the cost using the BFGS algorithm. """

    initial_theta = zeros((n, 1))

    final_theta, final_cost, _, _, _, _, _ = \
        optimize.fmin_bfgs(lambda t: cost(X, y, m, n, t, reg_factor),
                           initial_theta,
                           lambda t: gradient(X, y, m, n, t, reg_factor),
                           full_output=True, disp=False)

    return final_theta, final_cost


def accuracy(X, y, m, theta):
    """ Assigns a positive hypothesis if the sigmoid
        is greater than 0.5, and negative otherwise,
        and returns the accuracy of these hypotheses. """

    h = sigmoid(X.dot(theta))
    correct = 0

    for i in xrange(m):
        if h[i] > 0.5 and y[i, 0] == 1:
            correct += 1
        elif h[i] <= 0.5 and y[i, 0] == 0:
            correct += 1

    return correct * 100.0 / m


def main():
    """ Builds a regularized logistic regression model
        of the data including generated quadratic features,
        and measures its accuracy on the data. """

    data = np.genfromtxt('bbdm13.dat',
                          skip_header=6,
                          usecols=arange(2, 14, 1))

    m = data.shape[0]
    X = np.concatenate((ones((m, 1)),
                        data[:, 0:1],
                        data[:, 2:]), axis=1)
    n = X.shape[1]

    y = data[:, 1:2]

    reg_factor = 1.0

    X, n = generate_features(X, n)

    initial_theta = zeros((n, 1))
    final_theta, final_cost = logistic_regression(X, y, m, n, reg_factor)

    print "Accuracy before learning: %f" % accuracy(X, y, m, initial_theta)
    print "Accuracy after learning: %f" % accuracy(X, y, m, final_theta)

if __name__ == "__main__":
    main()
