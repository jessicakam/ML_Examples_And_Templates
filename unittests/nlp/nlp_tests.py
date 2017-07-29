#Date: 2017/07/29

import nlp
from unittest import TestCase
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
repo_dir = os.path.dirname(parentdir)
sys.path.insert(0,repo_dir) 

class TestNLPClasses(TestCase):
    def test_NLP(self):
        nlp = nlp.NLP()
        nlp.importDataset('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
        nlp.cleanTexts()
        nlp.createBagOfWordsModel()
        nlp.splitIntoTrainingAndTestSets(test_size=0.2, random_state=0)
        nlp.fitNaiveBayesToTrainingSet()
        nlp.predictResults()
        nlp.makeConfusionMatrix()