# 2017/08/09

from data_preprocessing import DataPreProcessing
from data_postprocessing import DataPostProcessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
class DimensionalityReduction(DataPreProcessing, DataPostProcessing):
    
    def __init__(self):
        super(DimensionalityReduction, self).__init__()
        
    def fitLogisticRegressionToTrainingSet(self, **kwargs):    
        self.classifier = LogisticRegression(**kwargs)
        self.classifier.fit(self.X_train, self.y_train)

    def visualizeResults(self, X_set, y_set, tuple_colors, title, xlabel, ylabel):
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, self.classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(tuple_colors))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(tuple_colors(i), label = j))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
        
    def visualizeTrainingSetResults(self, tuple_colors, title, xlabel, ylabel):
        self.visualizeResults(self.X_train, self.y_train, tuple_colors, title, xlabel, ylabel)
        
    def visualizeTestSet(self, tuple_colors, title, xlabel, ylabel):
        self.visualizeResults(self.X_test, self.y_test, tuple_colors, title, xlabel, ylabel)
        
        
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
class PCA_(DimensionalityReduction):
        
    def __init__(self):
        super(PCA, self).__init__()

    def applyPCA(self, **kwargs):
        pca = PCA(**kwargs)
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        self.explained_variance = pca.explained_variance_ratio_
    

from sklearn.decomposition import KernelPCA
class KernelPCA_(DimensionalityReduction):

    def __init__(self):
        super(KernelPCA_, self).__init__()

    def applyKernelPCA(self, **kwargs):
        self.kpca = KernelPCA(**kwargs)
        self.X_train = self.kpca.fit_transform(self.X_train)
        self.X_test = self.kpca.transform(self.X_test)
    
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
class LDA_():

    def __init__(self):
        super(LDA_, self).__init__()

    def applyLDA(self, **kwargs):
        self.lda = LDA(**kwargs)
        self.X_train = self.lda.fit_transform(self.X_train, self.y_train)
        self.X_test = self.lda.transform(self.X_test)
