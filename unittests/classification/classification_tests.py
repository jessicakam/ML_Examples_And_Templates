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
        self.assertTrue(logistic_regression.y.all(), instance + '.y should be set after importing dataset.')
        print("X: {0} \n".format(logistic_regression.X))
        print("y: {0} \n".format(logistic_regression.y))
        
        logistic_regression.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        self.assertTrue(logistic_regression.X_train, instance + '.X_train should be set after importing dataset')
        self.assertTrue(logistic_regression.X_test, instance + 'X_test should be set after importing dataset.')
        self.assertTrue(logistic_regression.y_train, instance + '.y_train should be set after importing dataset')
        self.assertTrue(logistic_regression.y_test, instance + '.y_test should be set after importing dataset.')
        print("X_train: {0} \n".format(logistic_regression.X_train))
        print("X_test: {0} \n".format(logistic_regression.X_test))
        print("y_train: {0} \n".format(logistic_regression.y_train))
        print("y_test: {0} \n".format(logistic_regression.y_test))
        
        logistic_regression.scaleFeatures2()
        self.assertTrue(logistic_regression.sc, instance + '.sc has not been set. X_train and X_test has not been scaled.')
        
        logistic_regression.fitToTrainingSet()
        self.assertTrue(logistic_regression.classifier, instance + '.classifier has not been set. Training set not fitted.')
        
        logistic_regression.predictTestResults()
        self.assertTrue(logistic_regression.y_pred, instance + '.y_pred has not been set. Results not predicted.')
        print("y_pred: {0} \n".format(logistic_regression.y_pred))
        
        logistic_regression.makeConfusionMatrix()
        self.assertTrue(self.cm, instance + '.cm not set. Confusion matrix not created.')
        print("cm: {0}".format(logistic_regression.cm))
        
        logistic_regression.visualizeTrainingsetResults('red', 'green', 'Age', 'Estimated Salary')
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
        knn.visualizeTrain('red', 'green', 'Age', 'Estimated Salary')
        knn.visualizeTest('red', 'green', 'Age', 'Estimated Salary')
    
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
        svm.visualizeTrain('red', 'green', 'Age', 'Estimated Salary')
        svm.visualizeTest('red', 'green', 'Age', 'Estimated Salary')
        
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
        k_svm.visualizeTrain('red', 'green', 'Age', 'Estimated Salary')
        k_svm.visualizeTest('red', 'green', 'Age', 'Estimated Salary')
    
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
        nb.visualizeTrain('red', 'green', 'Age', 'Estimated Salary')
        nb.visualizeTest('red', 'green', 'Age', 'Estimated Salary')
    
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
        decision_tree.visualizeTrain('red', 'green', 'Age', 'Estimated Salary')
        decision_tree.visualizeTest('red', 'green', 'Age', 'Estimated Salary')
    
    def test_RandomForestClassification(self):
        rand_forest = clf.RandomForestClassification()
        instance = 'rand_forest'
        self.assertEqual(rand_forest.name, 'RandomForestClassification', instance + '.name is incorrect.')
        
        rand_forest.importDataset2('Social_Network_Ads.csv', [2,3], 4)
        rand_forest.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        rand_forest.scaleFeatures()
        rand_forest.fitToTrainingSet(n_estimators=10, criterion=10, random_state=0)
        rand_forest.predictTestResults()
        rand_forest.makeConfusionMatrix()
        rand_forest.visualizeTrain('red', 'green', 'Age', 'Estimated Salary')
        rand_forest.visualizeTest('red', 'green', 'Age', 'Estimated Salary')

        
if __name__ == '__main__':
    unittest.main()