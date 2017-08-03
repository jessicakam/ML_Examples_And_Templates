# 2017/07/29

import unittest
from unittest import TestCase
import os,sys,inspect

# To get correct directory for file being tested
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir)
import nlp


class TestNLPClasses(TestCase):
    def test_NLP(self):
        nlp_inst = nlp.NLP()
        instance = 'nlp'
        
        nlp_inst.importDataset('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
        #self.assertTrue(nlp_inst.dataset.all(), instance + '.dataset should be set.')
        
        nlp_inst.cleanTexts(max_range=1000)
        self.assertTrue(nlp_inst.corpus, instance + '.corpus should not be empty.')
        
        nlp_inst.createBagOfWordsModel(max_features = 1500)
        self.assertTrue(nlp_inst.X.any(), instance + '.X should be set.')
        self.assertTrue(nlp_inst.y.any(), instance + '.y should be set.')
        
        nlp_inst.splitIntoTrainingAndTestSets(test_size=0.2, random_state=0)
        
        nlp_inst.fitNaiveBayesToTrainingSet()
        self.assertTrue(nlp_inst.classifier, instance + '.classifier should be set.')
        
        
        nlp_inst.predictResults()
        self.assertTrue(nlp_inst.y_pred.any(), instance + '.y_pred should be set.')
        
        nlp_inst.makeConfusionMatrix()
        self.assertTrue(nlp_inst.cm.all(), instance + '.cm should be created.')
        
        
if __name__ == '__main__':
    unittest.main()