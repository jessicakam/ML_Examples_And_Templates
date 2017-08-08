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

class TestDimensionalityReduction(TestCase):
    def test_PCA(self):
        pca = dr.PCA()
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
        lda = dr.LDA()
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
        k_pca = dr.KernelPCA()
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


if __name__ == '__main__':
    unittest.main()