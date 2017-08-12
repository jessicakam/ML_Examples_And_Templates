# 2017/08/09

from data_preprocessing import DataPreProcessing
from data_postprocessing import DataPostProcessing

import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
class XGBoost(DataPreProcessing, DataPostProcessing):
    
    def __init__(self):
        super(XGBoost, self).__init__()
        
    def fitToTrainingSet(self):
        self.classifier = XGBClassifier()
        self.classifier.fit(self.X_train, self.y_train)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import ListedColormap
class ModelSelection(DataPreProcessing, DataPostProcessing):
    
    def __init__(self):
        super(ModelSelection, self).__init__()
    
    def fitToTrainingSet(self, **kwargs):
        self.classifier = SVC(**kwargs)
        self.classifier.fit(self.X_train, self.y_train)
                
    def applyGridSearchToFindBestModels(self):
        parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                      {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
        grid_search = GridSearchCV(estimator = self.classifier,
                                   param_grid = parameters,
                                   scoring = 'accuracy',
                                   cv = 10,
                                   n_jobs = -1)
        self.grid_search = grid_search.fit(self.X_train, self.y_train)
        self.best_accuracy = self.grid_search.best_score_
        self.best_parameters = self.grid_search.best_params_
        
    def visualizeResults(self, X_set, y_set, title, xlabel, ylabel):
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, self.classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def visualizeTrainingSetResults(self, title, xlabel, ylabel):
        self.visualizeResults(self.X_train, self.y_train, title, xlabel, ylabel)
    
    def visualizeTestSetResults(self, title, xlabel, ylabel):
        self.visualizeResults(self.X_test, self.y_test, title, xlabel, ylabel)
