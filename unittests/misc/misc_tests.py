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
        instance = 'pca'
        
        pca.importDataset1('Wine.csv', 0, 13, 13)
        
        pca.splitInfoTrainingAndTestSets(test_size=0.2, random_state=0)
        
        pca.scaleFeatures2()
        
        pca.applyPCA(n_components=2)
        self.assertTrue(pca.explained_variance, instance + '.explained variance has not been set yet')
        
        pca.fitLogisticRegressionToTrainingSet(random_state=0)
        self.assertTrue(pca.classifier, instance + '.classifier has not been created.')
        
        pca.predictResults()
        
        pca.makeConfusionMatrix()
        
        pca.visualizeTrainingSetResults(tuple_colors = ('red', 'green', 'blue'), xlabel='PC1', ylabel='PC2')
        pca.visualizeTestSetResults(tuple_colors = ('red', 'green', 'blue'))

    def test_LDA(self):
        lda = misc.LDA()
        instance = 'lda'
        
        lda.importDataset('Wine.csv', 0, 13, 13)
        
        lda.splitInfoTrainingAndTestSets(test_size=0.2, random_state=0)
        
        lda.scaleFeatures2()
        
        lda.applyLDA(n_components=2)
        self.assertTrue(lda.lda, instance + '.lda has not been created yet.')
        
        lda.fitLogisticRegressionToTrainingSet()
        
        lda.predictResults()
        
        lda.makeConfusionMatrix()
        
        lda.visualizeTrainingSetResults(tuple_colors = ('red', 'green', 'blue'), xlabel='LD1', ylabel='LD2')
        lda.visualizeTestSetResults(tuple_colors = ('red', 'green', 'blue'))

    def test_KernelPCA(self):
        k_pca = misc.KernelPCA()
        instance = 'k_pca'
        
        k_pca.importDataset2('Social_Network_Ads.csv', [2, 3], 4)
        
        k_pca.splitInfoTrainingAndTestSets(test_size=0.25, random_state=0)
        
        k_pca.scaleFeatures2()
        
        k_pca.applyKernelPCA(n_components = 2, kernel = 'rbf')
        self.assertTrue(k_pca.kpca, instance + '.kpca has not been created.')
        
        k_pca.fitLogisticRegressionToTrainingSet()
        
        k_pca.predictResults()
        
        k_pca.makeConfusionMatrix()
        
        k_pca.visualizeTrainingSetResults(tuple_colors = ('red', 'green'), xlabel='Age', ylabel='Salary')
        k_pca.visualizeTestSetResults(tuple_colors = ('red', 'green'), xlabel='Age', ylabel='Salary')

class TestModelSelection(TestCase):
    def test_ModelSelection(self):
        gs = misc.GridSearch()
        instance = 'gs'
        
        gs.importDataset2('Social_Network_Ads.csv', [2, 3], 4)
        
        gs.splitIntoTrainingAndTestSets(test_size-0.25, random_state=0)
        
        gs.scaleFeatures2()
        
        gs.fitToTrainingSet(kernel='rbf', random_state=0)
        self.assertTrue(gs.classifier, instance + '.classifier has not been created.')
        
        gs.predictResults()
        
        gs.makeConfusionMatrix()
        
        gs.applyKFoldCrossValidation(cv=10)
        self.assertTrue(gs.accuracies, instance + '.accuracies has not been set.')
        self.assertTrue(gs.mean, instance + '.mean has not been set.')
        self.assertTrue(gs.std, instance + '.std has not been set.')
        
        gs.applyGridSearchToFindBestModels()
        self.assertTrue(gs.best_accuracy, instance + '.best_accuracy has not been set.')
        self.assertTrue(gs.best_parameters, instance + '.best_parameters has not been set.')
        
        gs.visualizeTrainingSetResults('Kernel SVM (Test set)', 'Age', 'Estimated Salary')
        gs.visualizeTestSetResults('Kernel SVM (Test set)', 'Age', 'Estimated Salary')

    # Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#
    def test_XGBoost(self):
        xgb = misc.XGBoost()
        instance = xgb
        
        xgb.importDataset1('Churn_Modelling.csv', 3, 13, 13)
        
        xgb.encodeCategoricalData()
        self.assertTrue(xbg.onehotencoder, instance + '.onehotencoder has not been created.')
        
        xgb.splitIntoTrainingAndTestSets(test_size=0.2, random_state=0)
        
        xgb.fitToTrainingSet()
        self.assertTrue(xbg.classifier, instance + '.classifier has not been created.')
        
        xgb.predictResults()
        
        xgb.makeConfusionMatrix()
        
        xgb.applyKFoldCrossValidation(estimator=self.classifier, X=self.X_train, y=self.y_train, cv=10)
        self.assertTrue(xgb.accuracies, instance + '.accuracies has not been set.')
        self.assertTrue(xgb.mean, instance + '.mean has not been set.')
        self.assertTrue(xgb.std, instance + '.std has not been set.')
    

class TestSOM(TestCase):
    def test_SOM(self):
        som = misc.SOM()
        instance = 'som'
        
        som.importDataset1('Credit_Card_Applications', 0, -1, -1)
        
        som.scaleFeatures()
        self.assertTrue(som.sc, instance + '.sc has not been created.')
        
        som.train(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
        self.assertTrue(som.som, instance + '.som has not been created.')
        
        som.visualizeResults()
        
        som.findFrauds()
        self.assertTrue(som.frauds, instance + '.frauds has not been found.')
        
class TestReducedBoltzmannMachines(TestCase):
    def test_ReducedBoltmannMachines(self):
        rbm = misc.ReducedBoltzmannMachines()
        instance = 'rbm'
        
        rbm.importDataset4()
        self.assertTrue(rbm.movies, instance + '.movies has not been set.')
        self.assertTrue(rbm.users, instance + '.users has not been set.')
        self.assertTrue(rbm.ratings, instance + '.ratings has not been set.')
        
        rbm.prepareTrainingAndTestSets()
        self.assertTrue(rbm.training_set, instance + '.training_set has not been set.')
        self.assertTrue(rbm.test_set, instance + '.test_set has not been set.')
        self.assertTrue(rbm.nb_users, instance + '.nb_users has not been set.')
        self.assertTrue(rbm.nb_movies, instance + '.nb_movies has not been set.')
        
        rbm.convertData()
        self.assertTrue(rbm.new_data, instance + '.new_data has not been set.')
        
        training_set_before = rbm.training_set
        test_set_before = rbm.test_set
        rbm.convertIntoTensors()
        self.assertTrue(training_set_before != rbm.training_set, instance + '.training_set not a tensor.')
        self.assertTrue(test_set_before != rbm.test_set, instance + '.test_set not a tensor.')
        
        rbm.convertIntoBinary()
        self.assertTrue(rbm.training_set.all(a==0 or a==1 or a==-1), instance + '.training_set should be binary')
        self.assertTrue(rbm.test_set.all(a==0 or a==1 or a==-1), instance + '.test_set should be binary')
        
        rbm.createNN(nv=len(rbm.training_set[0]), nh=100)
        
        rbm.train(mb_epoch=10)
        # screen printouts
        
        rbm.test()
        # screen printouts

class AutoEncoder(TestCase):
    def test_AutoEncoder(self):
        ae = misc.AutoEncoder()
        instance = 'ae'
        
        ae.importDataset4()
        
        ae.prepareTrainingAndTestSets()
        
        ae.convertData()
        
        ae.convertIntoTensors()
        
        ae.createNN()
        self.assertTrue(ae.sae, instance + '.sae has not been set.')
        self.assertTrue(ae.criterion, instance + '.criterion has not been set.')
        self.assertTrue(ae.optimizer, instance + '.optimizer has not been set.')
        
        ae.train(nb_epoch=200)
        # screen printouts
        
        ae.test()
        # screen printouts
    
if __name__ == '__main__':
    unittest.main()