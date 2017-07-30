#Date: 2017/07/29

import misc
from unittest import TestCase
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir) 

class TestDimensionalityReduction(TestCase):
    def test_PCA(self):
        pca = misc.PCA()
        pca.importDataset('Wine.csv', 0, 13, 13)
        pca.splitInfoTrainingAndTestSets(test_size=0.2, random_state=0)
        pca.scaleFeatures()
        pca.applyPCA()
        pca.fitLogisticRegressionToTrainingSet()
        pca.predictResults()
        pca.makeConfusionMatrix()
        pca.visualizeTrainingSetResults(tuple_colors = ('red', 'green', 'blue'), xlabel='PC1', ylabel='PC2')
        pca.visualizeTestSetResults(tuple_colors = ('red', 'green', 'blue'))

    def test_LDA(self):
        lda = misc.LDA()
        lda.importDataset('Wine.csv', 0, 13, 13)
        lda.splitInfoTrainingAndTestSets(test_size=0.2, random_state=0)
        lda.scaleFeatures()
        lda.applyLDA()
        lda.fitLogisticRegressionToTrainingSet()
        lda.predictResults()
        lda.makeConfusionMatrix()
        lda.visualizeTrainingSetResults(tuple_colors = ('red', 'green', 'blue'), xlabel='LD1', ylabel='LD2')
        lda.visualizeTestSetResults(tuple_colors = ('red', 'green', 'blue'))

    def test_KernelPCA(self):
        k_pca = misc.KernelPCA()
        k_pca.importDataset('Social_Network_Ads.csv', lst_columns=[2, 3], 4)
        k_pca.splitInfoTrainingAndTestSets(test_size=0.25, random_state=0)
        k_pca.scaleFeatures()
        k_pca.applyKernelPCA() #
        k_pca.fitLogisticRegressionToTrainingSet()
        k_pca.predictResults()
        k_pca.makeConfusionMatrix()
        k_pca.visualizeTrainingSetResults(tuple_colors = ('red', 'green'), xlabel='Age', ylabel='Salary')
        k_pca.visualizeTestSetResults(tuple_colors = ('red', 'green'), xlabel='Age', ylabel='Salary')

class TestModelSelection(TestCase):
    def test_ModelSelection(self):
        gs = misc.GridSearch()
        gs.importDataset('Social_Network_Ads.csv', lst_columns=[2, 3], 4)
        gs.splitIntoTrainingAndTestSets(test_size-0.25, random_state=0)
        gs.scaleFeatures() #
        gs.fitToTrainingSet(kernel='rbf', random_state=0) #
        gs.predictResults() #
        gs.makeConfusionMatrix() #
        gs.applyKFoldCrossValidation(estimator=self.classifier, X=self.X_train, y=self.y_train, cv=10) #
        gs.applyGridSearchToFindBestModels() #
        gs.visualizeTrainingSetResults() ##
        gs.visualizeTestSetResults() ##

    # Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#
    def test_XGBoost(self):
        xgb = misc.XGBoost()
        xgb.importDataset('Churn_Modelling.csv', 3, 13, 13)
        xgb.encodeCategoricalData(1) ##same thing as example?
        xgb.encodeCategoricalData(2) ##
        xgb.splitIntoTrainingAndTestSets(test_size=0.2, random_state=0)
        xgb.fitToTrainingSet()
        xgb.predictResults()
        xgb.makeConfusionMatrix()
        xgb.applyKFoldCrossValidation(estimator=self.classifier, X=self.X_train, y=self.y_train, cv=10)
    
#####can add more stuff to DataPreprocessing, just that everything that inherit from it not have to use it...
#####can also try adding another class with stuff common between most ml classes


class SOM(DataPreprocessing):
    pass

class BoltzmannMachines(DataPreprocessing):
    pass

class AutoEncoder(DataPreprocessing):
    pass