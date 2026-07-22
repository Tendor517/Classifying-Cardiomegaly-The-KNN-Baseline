#filename: knn.py
import numpy as np
from collections import Counter

class KNearestNeighbor:
    def __init__(self):
        # KNN has no learnable parameters. 
        pass

    def train(self, X, y):
        """
        Training KNN is simply memorizing the training data.
        X shape: (num_train, features)
        y shape: (num_train,)
        """
        self.X_train = X
        self.y_train = y

    def compute_distances(self, X_test):
        """
        Computes the L2 distance between all test and training data 
        using full vectorization (0 loops).
        
        We use the mathematical expansion: (x - y)^2 = x^2 - 2xy + y^2
        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        
        # 1. x^2: Square the test data and sum across features
        # Shape: (num_test, 1)
        test_sum = np.sum(np.square(X_test), axis=1, keepdims=True)
        
        # 2. y^2: Square the training data and sum across features
        # Shape: (num_train,)
        train_sum = np.sum(np.square(self.X_train), axis=1)
        
        # 3. -2xy: The dot product of test and transposed training data
        # Shape: (num_test, num_train)
        inner_product = np.dot(X_test, self.X_train.T)
        
        # Combine and take the square root
        # Broadcasting automatically expands test_sum and train_sum to (num_test, num_train)
        dists = np.sqrt(test_sum - 2 * inner_product + train_sum)
        
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances, predict labels for each test image 
        by finding the k nearest neighbors and taking a majority vote.
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        
        for i in range(num_test):
            # argsort returns the indices that would sort the array
            # We slice the first 'k' indices (the closest training images)
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            
            # Count the occurrences of each label and grab the most common one
            vote = Counter(closest_y)
            y_pred[i] = vote.most_common(1)[0][0]
            
        return y_pred

    def predict(self, X_test, k=1):
        """Wrapper function to compute distances and predict labels."""
        dists = self.compute_distances(X_test)
        return self.predict_labels(dists, k=k)