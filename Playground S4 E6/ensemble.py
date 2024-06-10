import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Ensemble():
    def __init__(self, models):
        self.models = models
        self.weights = []
        self.individual_model_scores = []

    def calc_linear_weights(self, min_weight, max_weight):
        """
        scores is a list of scores of each model in the ensemble.
        min_weight is the weight assigned to the worst performing model.
        max_weight is the weight assigned to the best performing model.
        """
        assert(len(self.individual_model_scores) > 0)
        self.weights = []
        sum = 0
        for score in self.individual_model_scores:
            weight = (min_weight + ((score - min(self.individual_model_scores)) / 
                                   (max(self.individual_model_scores) - min(self.individual_model_scores))) * max_weight)
            sum += weight
            self.weights.append(weight)
        for i in range(len(self.weights)):
            self.weights[i] /= sum
    
    def score_individual_models(self, X, y):
        for model in self.models:
            print("model score", accuracy_score(y, model.predict(X)))
            self.individual_model_scores.append(accuracy_score(y, model.predict(X)))
    
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
    
    def predict(self, X):
        if len(self.weights) == 0:
            return None
        
        model_preds = []
        for model in self.models:
            model_preds.append(model.predict(X))

        ensemble_pred = []
        for i in range(X.shape[0]):
            votes = dict()
            best_vote = 0
            voted_class = None
            for j in range(len(self.models)):
                pred = model_preds[j][i]
                if pred not in votes:
                    votes[pred] = self.weights[j]
                else:
                    votes[pred] += self.weights[j]
                if votes[pred] > best_vote:
                    voted_class = pred
                    best_vote = votes[pred]
            assert(voted_class is not None)
            ensemble_pred.append(voted_class)

        return ensemble_pred