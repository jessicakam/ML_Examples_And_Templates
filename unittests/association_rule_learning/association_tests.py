# 2017/07/29

import unittest
from unittest import TestCase
import os,sys,inspect

# To get correct directory for file being tested
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir) 
import association_rule_learning as assoc_rule_learn


class TestAssociationRuleLearningClasses(TestCase):
    def test_Apriori(self):
        apriori = assoc_rule_learn.Apriori()
        self.assertEqual(apriori.X, None, 'apriori.X should be None.')
        self.assertEqual(apriori.y, None, 'apriori.y should be None.')

        apriori.preprocessData('Market_Basket_Optimisation.csv', header=None, num_lines=7500, range_max=20,)
        self.assertFalse(apriori.dataset.empty, 'apriori.dataset should not be empty.')

        apriori.trainOnDataset(min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)
        self.assertTrue(apriori.rules, 'apriori should have a rules attribute')

        results = apriori.visualizeResults()
        self.assertTrue(results, 'apriori should return results (a list of rules).')
 
        
if __name__ == '__main__':
    unittest.main()
