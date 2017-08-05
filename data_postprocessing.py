# 2017/08/01

from sklearn.metrics import confusion_matrix

class DataPostProcessing():
    
    def predictResults(self):
        self.y_pred = self.classifier.predict(self.X_test)
        
    def makeConfusionMatrix(self):
        self.cm = confusion_matrix(self.y_test, self.y_pred)
