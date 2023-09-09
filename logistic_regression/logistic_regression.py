import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self,y):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.weights = []
        self.bias = []
        
    def fit(self, X, y, learning_rate, iterations):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        self.y_true = y
        cost_history = []
        n_features = X.shape[1]  # Change from X.shape[0] to X.shape[1]
        m_samples = X.shape[0]  # Change from X.shape[1] to X.shape[0]

        weights = np.zeros((n_features, 1))
        B = 0

        # Reshape y to match A's shape
        y = y.values.reshape(m_samples, 1)

        for i in range(iterations):
            Z = np.dot(X, weights) + B  # Swap X and weights in dot product
            A = sigmoid(Z)

            cost = -(1 / m_samples) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
            dWeights = (1 / m_samples) * np.dot(X.T, A - y)  # Swap X and A in dot product
            dB = (1 / m_samples) * np.sum(A - y)

            weights = weights - learning_rate * dWeights
            B = B - learning_rate * dB
            cost_history.append(cost)

            if (i % (iterations / 10) == 0):
                print('cost after iteration {}: {}'.format(i, cost))

        self.weights = weights
        self.bias = B

        
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        Z = np.dot(X, self.weights) + self.bias  # Swap self.weights and X in dot product
        return sigmoid(Z).flatten()

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    (Minimizing this is equivalent to maximizing the likelihood of the predictions)
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        