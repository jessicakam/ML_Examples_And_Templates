"""
Name: Jessica Kam
Date: 2017/07/28
"""
import classification as clf
from unittest import TestCase
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir) 

class TestClassificationClasses(TestCase):
    
    def test_LogisticRegression(self):
        logistic_regression = clf.LogisticRegression()
        logistic_regression.importDataset('Social_Network_Ads.csv', list_of_columns=[2,3], 4)
        logistic_regression.splitIntoTrainingAndTestSets(test_size=0.25, random_state=0)
        logistic_regression.scaleFeatures()
        logistic_regression.fitToTrainingSet()
        logistic_regression.predictTestResults()
        logistic_regression.makeConfusionMatrix()
        logistic_regression.visualizeTrainingsetResults() #
        logistic_regression.visualizeTestSetResults() #
    
    def test_KNN(self):
        knn = clf.KNN()
        knn.importDataset('Social_Network_Ads.csv', list_of_columns=[2,3], 4)
        knn.splitIntoTrainingTestSet(test_size=0.25, random_state=0)
        knn.scaleFeatures()
        knn.fitToTrainingSet()
        knn.predictTestResults()
        knn.makeConfusionMatrix()
        knn.visualizeTrain()#
        knn.visualizeTest() #
    
    def test_SVM(self):
        svm = clf.SVM()
        svm.importDataset('Social_Network_Ads.csv', list_of_columns=[2,3], 4)
        svm.splitIntoTrainingTestSet(test_size=0.25, random_state=0)
        svm.scaleFeatures()
        svm.fitToTrainingSet()
        svm.predictTestResults()
        svm.makeConfusionMatrix()
        svm.visualizeTrain()#
        svm.visualizeTest() #
        
    
    def test_KernelSVM(self):
        k_svm = clf.KernelSVM()
        k_svm.importDataset('Social_Network_Ads.csv', list_of_columns=[2,3], 4)
        k_svm.splitIntoTrainingTestSet(test_size=0.25, random_state=0)
        k_svm.scaleFeatures()
        k_svm.fitToTrainingSet(random_state=0)
        k_svm.predictTestResults()
        k_svm.makeConfusionMatrix()
        k_svm.visualizeTrain()#
        k_svm.visualizeTest() #
    
    def test_NaiveBayes(self):
        k_svm = clf.KernelSVM()
        k_svm.importDataset('Social_Network_Ads.csv', list_of_columns=[2,3], 4)
        k_svm.splitIntoTrainingTestSet(test_size=0.25, random_state=0)
        k_svm.scaleFeatures()
        k_svm.fitToTrainingSet(random_state=0)
        k_svm.predictTestResults()
        k_svm.makeConfusionMatrix()
        k_svm.visualizeTrain()#
        k_svm.visualizeTest() #
    
    def test_DecisionTreeClassification(self):
        decision_tree = clf.DecisionTreeClassification()
        decision_tree.importDataset('Social_Network_Ads.csv', list_of_columns=[2,3], 4)
        decision_tree.splitIntoTrainingTestSet(test_size=0.25, random_state=0)
        decision_tree.scaleFeatures()
        decision_tree.fitToTrainingSet(criterion=entropy, random_state=0)
        decision_tree.predictTestResults()
        decision_tree.makeConfusionMatrix()
        decision_tree.visualizeTrain()#
        decision_tree.visualizeTest() #
    
    def test_RandomForestClassification(self):
        rand_forest = clf.RandomForestClassification()
        rand_forest.importDataset('Social_Network_Ads.csv', list_of_columns=[2,3], 4)
        rand_forest.splitIntoTrainingTestSet(test_size=0.25, random_state=0)
        rand_forest.scaleFeatures()
        rand_forest.fitToTrainingSet(n_estimators=10, criterion=10, random_state=0)
        rand_forest.predictTestResults()
        rand_forest.makeConfusionMatrix()
        rand_forest.visualizeTrain()#
        rand_forest.visualizeTest() #

if __name__ == '__main__':
    unittest.main()