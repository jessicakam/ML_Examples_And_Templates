"""
Name: Jessica Kam
Date: 2017/07/28
"""
import classification
from unittest import TestCase
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir) 

class TestClassificationClasses(TestCase):
    
    def test_SimpleLinearRegression(self):
        pass
    
    def test_LogisticRegression(self):
        pass
    
    def test_KNN(self):
        pass
    
    def test_SVM(self):
        pass
    
    def test_KernelSVM(self):
        pass
    
    def test_NaiveBayes(self):
        pass
    
    def test_DecisionTreeClassification(self):
        pass
    
    def test_RandomForestClassification(self):
        pass

if __name__ == '__main__':
    unittest.main()