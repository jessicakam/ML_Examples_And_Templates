# 2017/07/29

import unittest
from unittest import TestCase
import os,sys,inspect

# To get correct directory for file being tested
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir)
import misc

class TestDimensionalityReduction(TestCase):
    def test_PCA(self):
        pca = misc.PCA()
        #instance = 'logistic_regression'
        #self.assertEqual(logistic_regression.X, None, instance + '.X should be None.')
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
##2 different imports in datapreprocess along with diff versions for other things?
##add csv files for everything in misc
##remove my name for all files, change date -> date started
##look up general way to instantiate
##have singleton or something where have bunch possible functions but choose correct one based on name -- DECORATORS???
class TestSOM(TestCase):
    def test_SOM(self):
        som = misc.SOM()
        som.importDataset('Credit_Card_Applications', 0, -1, -1)
        som.scaleFeatures()
        som.train()
        som.visualizeResults()
        som.findFrauds()
        
class TestBoltzmannMachines(TestCase):
    def test_ReducedBoltmannMachines(self):
        rbm = misc.ReducedBoltzmannMachines()
        rbm.importDataset() #
        rbm.prepareTrainingAndTestSets() #
        rbm.convertData()
        rbm.convertIntoTensors()
        rbm.convertIntoBinary()
        rbm.createNN()
        rbm.train()
        rbm.test()
        

class AutoEncoder(TestCase):
    def test_AutoEncoder(self):
        ae = misc.AutoEncoder()
        ae.importDataset() #
        ae.prepareTrainingAndTestSets()
        ae.convertData()
        ae.convertIntoTensors()
        ae.createNN()
        ae.train()
        ae.test()
        
    
if __name__ == '__main__':
    unittest.main()