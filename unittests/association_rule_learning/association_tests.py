"""
Date: 2017/07/29
"""
import association_rule_learning as assoc_rule_learn
from unittest import TestCase
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir) 

class TestAssociationRuleLearningClasses(TestCase):
    
    def test_Apriori(self):
        apriori = assoc_rule_learn.Apriori()
        apriori.preprocessData('Market_Basket_Optimization.csv', header=None, num_observations=7501, range_max=20)
        apriori.trainOnDataset(min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)
        apriori.visualizeResults()