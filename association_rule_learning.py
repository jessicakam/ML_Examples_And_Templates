# 2017/07/29

from data_preprocessing import DataPreProcessing
import pandas as pd

class AssociationRuleLearning(DataPreProcessing):
    def __init__(self):
        super(AssociationRuleLearning, self).__init__()
        
from apyori import apriori
class Apriori(AssociationRuleLearning):
    def __init__(self):
        super(Apriori, self).__init__()
        
    def preprocessData(self, csv_file, header, num_lines, range_max):
        self.dataset = pd.read_csv(csv_file, header, engine='python')
        self.transactions = []
        for i in range(0, num_lines):
            self.transactions.append([str(self.dataset.values[i,j]) for j in range(0, range_max)])
    
    def trainOnDataset(self, **kwargs):
        self.rules = apriori(self.transactions, **kwargs)
    
    def visualizeResults(self):
        self.results = list(self.rules)
        return self.results