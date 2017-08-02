# 2017/08/01

from sklearn.metrics import confusion_matrix

class DataPostProcessing():
    def __init__(self):
        pass
    
    def makeConfusionMatrix(self):
        self.cm = confusion_matrix(self.y_test, self.y_pred) 


