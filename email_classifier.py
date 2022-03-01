#######################################################
#                 ML Assignment 2                     #
#              Dvizma Sinha, CS20M504                 #
#                CS5011W, 25/05/2021                  #
#######################################################

from feature_extractor import FeatureExtractor
from enum import IntEnum
import numpy as np
import mathlib
from pathlib import Path


class EmailClass(IntEnum):
    NON_SPAM = 0
    SPAM = 1


class LogisticRegression:

    weights_path = Path(__file__).parent.resolve() / "lrweight.npy"

    def __init__(self):
        self.weights = None

    def sigmoid(self, x):
        """Calculate sigmoid for each element of array x"""
        return 1 / (1 + mathlib.exp(-x))

    def fit(self, features, labels, learning_rate=0.05, num_iterations=2000):
        """Run gradient descent to calculate weights for mapping features to labels

        Args:
            features (np.array): features extracted from emails
            labels (np.array): known labels on the features
            learning_rate (float, optional): Learning rate of model. Defaults to 0.05.
            num_iterations (int, optional): Number of iteration for gradient descent. Defaults to 2000.
        """
        num_points, num_features = features.shape
        wdg = np.random.rand(num_features, 1)

        for i in range(num_iterations):
            y_est = mathlib.matmul(features, wdg)
            y_est = self.sigmoid(y_est)
            gradient = np.matmul(mathlib.transpose(features), y_est - labels)
            wdg = wdg - learning_rate * gradient / num_points

        self.weights = wdg
        np.save(self.weights_path, self.weights, allow_pickle=True)

    def predict(self, features):
        """Predict labels for features

        Args:
            features (np.array): test features

        Returns:
            np.array: test labels
        """
        self.weights = np.load(
            self.weights_path, allow_pickle=True).reshape(-1, 1)
        y_est = mathlib.matmul(features, self.weights)
        y_est = self.sigmoid(y_est)
        result = (y_est >= 0.5).astype(np.int32)

        return result


class EmailClassifier:

    @staticmethod
    def train(non_spam_path, spam_path):
        """Extract features from spam and non spam emails. Train the classifier

        Args:
            non_spam_path (Path): Path to non spam emails
            spam_path (Path): Path to spam emails
        """
        # Get file paths for spam and non spam emails
        non_spam_files = list(Path(non_spam_path).glob("*.txt"))
        spam_files = list(Path(spam_path).glob("*.txt"))

        print("Non Spam files:", len(non_spam_files))
        print("Spam Files: ", len(spam_files))

        # Build feature vocabulary
        print("Building vocabulary")
        feature_extractor = FeatureExtractor()
        feature_extractor.build_volcabulary(non_spam_files + spam_files)

        # Extract features from training data
        print("Extracting features")
        features = feature_extractor.extract_features(log=True)

        # Set lables
        labels = np.array([0] * len(non_spam_files) + [1] * len(spam_files))

        print("Training the model")
        logreg = LogisticRegression()
        logreg.fit(features, labels, num_iterations=4000, learning_rate=0.04)

    @staticmethod
    def test(dir):
        """Test model using files from a give directory and generate report

        Args:
            dir (Path): Path to test email
        """
        logreg = LogisticRegression()
        test_files = list(Path(dir).glob("*.txt"))
        test_files.sort()
        feature_extractor = FeatureExtractor()
        feature_extractor.read_emails(test_files)
        x_test = feature_extractor.extract_features(log=False)
        y_pred = list(logreg.predict(x_test).reshape(-1,))

        print("*********************Classification Report***********************")
        print("{:<25}{:<25}{:10}".format("FileName",
                                         "Predicted Class", "Predicted Value"))
        for r, x in zip(y_pred, test_files):
            print(f"{x.name : <25}{EmailClass(r).name:<25}{r:10}")
