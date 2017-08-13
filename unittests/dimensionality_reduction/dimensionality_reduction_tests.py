# 2017/08/06

import unittest
from unittest import TestCase
import os,sys,inspect

# To get correct directory for file being tested
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir)
import dimensionality_reduction as dr

class TestDimensionalityReductionClasses(TestCase):
    def test_PCA(self):
        pca = dr.PCA_()
        instance = 'pca'
        
        pca.importDataset1('Wine.csv', 0, 13, 13)
        
        pca.splitIntoTrainingAndTestSets(test_size=0.2, random_state=0)
        
        pca.scaleFeatures2()
        
        pca.applyPCA(n_components=2)
        self.assertTrue(pca.explained_variance.any(), instance + '.explained variance has not been set yet')
        
        pca.fitLogisticRegressionToTrainingSet(random_state=0)
        self.assertTrue(pca.classifier, instance + '.classifier has not been created.')
        
        pca.predictResults()
        
        pca.makeConfusionMatrix()
        
        pca.visualizeTrainingSetResults(tuple_colors = ('red', 'green', 'blue'), title='PCA Logistic Regression (Training Set)', xlabel='PC1', ylabel='PC2')
        pca.visualizeTestSetResults(tuple_colors = ('red', 'green', 'blue'), title='PCA Logistic Regression (Test Set)', xlabel='PC1', ylabel='PC2')
    

    def test_LDA(self):
        lda = dr.LDA_()
        instance = 'lda'
        
        lda.importDataset1('Wine.csv', 0, 13, 13)
        
        lda.splitIntoTrainingAndTestSets(test_size=0.2, random_state=0)
        
        lda.scaleFeatures2()
        
        lda.applyLDA(n_components=2)
        self.assertTrue(lda.lda, instance + '.lda has not been created yet.')
        
        lda.fitLogisticRegressionToTrainingSet()
        
        lda.predictResults()
        
        lda.makeConfusionMatrix()
        
        lda.visualizeTrainingSetResults(tuple_colors = ('red', 'green', 'blue'), title='LDA Logistic Regression (Training Set)', xlabel='PC1', ylabel='PC2')
        lda.visualizeTestSetResults(tuple_colors = ('red', 'green', 'blue'), title='LDA Logistic Regression (Test Set)', xlabel='PC1', ylabel='PC2')
    

    def test_KernelPCA(self):
        k_pca = dr.KernelPCA_()
        instance = 'k_pca'
        
        k_pca.importDataset2('Social_Network_Ads.csv', [2, 3], 4)
        
        k_pca.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        
        k_pca.scaleFeatures2()
        
        k_pca.applyKernelPCA(n_components = 2, kernel = 'rbf')
        self.assertTrue(k_pca.kpca, instance + '.kpca has not been created.')
        
        k_pca.fitLogisticRegressionToTrainingSet()
        
        k_pca.predictResults()
        
        k_pca.makeConfusionMatrix()
        
        k_pca.visualizeTrainingSetResults(tuple_colors = ('red', 'green'), title='KernelPCA Logistic Regression (Training Set)', xlabel='Age', ylabel='Salary')
        k_pca.visualizeTestSetResults(tuple_colors = ('red', 'green'), title='KernelPCA Logisitic Regression (Test Set)', xlabel='Age', ylabel='Salary')


if __name__ == '__main__':
    unittest.main()