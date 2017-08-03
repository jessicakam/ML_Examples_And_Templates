# 2017/07/28

import unittest
from unittest import TestCase
import os,sys,inspect

# To get correct directory for file being tested
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir)
import classification as clf


class TestClassificationClasses(TestCase):        
    def test_LogisticReg(self):
        logistic_regression = clf.LogisticReg()
        instance = 'logistic_regression'
        self.assertEqual(logistic_regression.X, None, instance + '.X should be None.')
        self.assertEqual(logistic_regression.y, None, instance + '.y should be None.')
        self.assertEqual(logistic_regression.name, 'LogisticReg', instance + '.name is incorrect.')
        
        logistic_regression.importDataset2('Social_Network_Ads.csv', [2,3], 4)
        self.assertTrue(logistic_regression.X.all(), instance + '.X should be set after importing dataset')
        self.assertTrue(logistic_regression.y.any(), instance + '.y should be set after importing dataset.')
        
        logistic_regression.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        self.assertTrue(logistic_regression.X_train.all(), instance + '.X_train should be set after importing dataset')
        self.assertTrue(logistic_regression.X_test.all(), instance + 'X_test should be set after importing dataset.')
        self.assertTrue(logistic_regression.y_train.any(), instance + '.y_train should be set after importing dataset')
        self.assertTrue(logistic_regression.y_test.any(), instance + '.y_test should be set after importing dataset.')
        
        logistic_regression.scaleFeatures2()
        self.assertTrue(logistic_regression.sc, instance + '.sc has not been set. X_train and X_test has not been scaled.')
        
        logistic_regression.fitToTrainingSet()
        self.assertTrue(logistic_regression.classifier, instance + '.classifier has not been set. Training set not fitted.')
        
        logistic_regression.predictTestResults()
        self.assertTrue(logistic_regression.y_pred.any(), instance + '.y_pred has not been set. Results not predicted.')
        
        logistic_regression.makeConfusionMatrix()
        self.assertTrue(logistic_regression.cm.all(), instance + '.cm not set. Confusion matrix not created.')
         
        logistic_regression.visualizeTrainingSetResults('red', 'green', 'Age', 'Estimated Salary')
        logistic_regression.visualizeTestSetResults('red', 'green', 'Age', 'Estimated Salary')
    
    def test_KNN(self):
        knn = clf.KNN()
        instance = 'knn'
        self.assertEqual(knn.name, 'KNN', instance + '.name is incorrect.')
        
        knn.importDataset2('Social_Network_Ads.csv', [2,3], 4)        
        knn.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        knn.scaleFeatures2()
        knn.fitToTrainingSet()
        knn.predictTestResults()
        knn.makeConfusionMatrix()
        knn.visualizeTrainingSetResults('red', 'green', 'Age', 'Estimated Salary')
        knn.visualizeTestSetResults('red', 'green', 'Age', 'Estimated Salary')
    
    def test_SVM(self):
        svm = clf.SVM()
        instance = 'svm'
        self.assertEqual(svm.name, 'SVM', instance + '.name is incorrect.')
        
        svm.importDataset2('Social_Network_Ads.csv', [2,3], 4)        
        svm.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        svm.scaleFeatures2()
        svm.fitToTrainingSet()
        svm.predictTestResults()
        svm.makeConfusionMatrix()
        svm.visualizeTrainingSetResults('red', 'green', 'Age', 'Estimated Salary')
        svm.visualizeTestSetResults('red', 'green', 'Age', 'Estimated Salary')
        
    def test_KernelSVM(self):
        k_svm = clf.KernelSVM()
        instance = 'k_svm'
        self.assertEqual(k_svm.name, 'KernelSVM', instance + '.name is incorrect.')
        
        k_svm.importDataset2('Social_Network_Ads.csv', [2,3], 4)
        k_svm.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        k_svm.scaleFeatures2()
        k_svm.fitToTrainingSet(random_state=0)
        k_svm.predictTestResults()
        k_svm.makeConfusionMatrix()
        k_svm.visualizeTrainingSetResults('red', 'green', 'Age', 'Estimated Salary')
        k_svm.visualizeTestSetResults('red', 'green', 'Age', 'Estimated Salary')
    
    def test_NaiveBayes(self):
        nb = clf.NaiveBayes()
        instance = 'nb'
        self.assertEqual(nb.name, 'NaiveBayes', instance + '.name is incorrect.')
        
        nb.importDataset2('Social_Network_Ads.csv', [2,3], 4)
        nb.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        nb.scaleFeatures2()
        nb.fitToTrainingSet(random_state=0)
        nb.predictTestResults()
        nb.makeConfusionMatrix()
        nb.visualizeTrainingSetResults('red', 'green', 'Age', 'Estimated Salary')
        nb.visualizeTestSetResults('red', 'green', 'Age', 'Estimated Salary')
    
    def test_DecisionTreeClassification(self):
        decision_tree = clf.DecisionTreeClassification()
        instance = 'decision_tree'
        self.assertEqual(decision_tree.name, 'DecisionTreeClassification', instance + '.name is incorrect.')
       
        decision_tree.importDataset2('Social_Network_Ads.csv', [2,3], 4)
        decision_tree.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        decision_tree.scaleFeatures2()
        decision_tree.fitToTrainingSet(criterion='entropy', random_state=0)
        decision_tree.predictTestResults()
        decision_tree.makeConfusionMatrix()
        decision_tree.visualizeTrainingSetResults('red', 'green', 'Age', 'Estimated Salary')
        decision_tree.visualizeTestSetResults('red', 'green', 'Age', 'Estimated Salary')
    
    def test_RandomForestClassification(self):
        rand_forest = clf.RandomForestClassification()
        instance = 'rand_forest'
        self.assertEqual(rand_forest.name, 'RandomForestClassification', instance + '.name is incorrect.')
        
        rand_forest.importDataset2('Social_Network_Ads.csv', [2,3], 4)
        rand_forest.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        rand_forest.scaleFeatures2()
        rand_forest.fitToTrainingSet(n_estimators=10, criterion='entropy', random_state=0)
        rand_forest.predictTestResults()
        rand_forest.makeConfusionMatrix()
        rand_forest.visualizeTrainingSetResults('red', 'green', 'Age', 'Estimated Salary')
        rand_forest.visualizeTestSetResults('red', 'green', 'Age', 'Estimated Salary')

        
if __name__ == '__main__':
    unittest.main()