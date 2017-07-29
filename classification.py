"""
Name: Jessica Kam
Date: 2017/07/28
"""

from data_preprocessing import DataPreprocessing
from sklearn,metrics import confusion_matrix
from matplotlib.colors import ListedColormap


class Classification(DataPreprocessing, list_of_columns):
    def __init__(self):
        pass #
        
    def importDataset(self, csv_file):
        self.dataset = pd.read_csv(csv_file)
        self.X = dataset.iloc[: list_of_columns].values
        self.y = dataset.iloc[:, y_index].values
    
    def scaleFeatures(self):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        
    def fitToTrainingSet(self):
        pass
    
    def makeConfusionMatrix(self):
       self.cm = confusion_matrix(self.y_test, self.y_pred) 

    def visualizeTrainingSetResults(self):
        pass #
    
    def visualizeTestSetResults(self):
        pass #

from sklearn.linear_model import LogisticRegression
class LogisticRegression(Classification):
    def fitToTrainingSet(self, **kwargs):
        self.classifier = LogisticRegression(**kwargs)
        self.classifier.fit(self.X_train, self.y_train)
    
    def predictTestResults(self):
        self.y_pred = self.classifier.predict(self.X_test)
        
    def visualizeTrainingSetResults(self):
        pass #
    
    def visualizeTestSetResults(self):
        pass #

form sklearn.neighbors import KNeighborsClassifier
class KNN(Classification):
    def fitToTrainingSet(self, **kwargs):
        self.classifier = KNeighborsClassifier(**kwargs)
        self.classifier.fit(self.X_train, self.y_train)

    def visualizeTrain():
        pass #
    def visualizeTest():
        pass #
        
from sklearn.svm import SVC
class SVM(Classification):
    def fitToTrainingSet(self, **kwargs):
        self.classifier = SVC(**kwargs)
        self.classifier.fit(self.X_train, self.y_train)


class KernelSVM(SVM):
    def fitToTrainingSet(self, **kwargs):
        super(KernelSVM, self).fitToTrainingSet(kernel='rbf', **kwargs)

from sklearn.naive_bayes import GaussianNB
class NaiveBayes(Classification):
    def fitToTrainingSet(self, **kwargs):
        self.classifer = GaussianNB()
        self.classifer.fit(self.X_train, self.y_train)

from sklearn.tree import DecisionTreeClassifier
class DecisionTreeClassification(Classification):
    def fitToTrainingSet(self, **kwargs):
        self.classifier = DecisionTreeClassifier(**kwargs)
        self.classifier.fit(self.X_train, self.y_train)

from sklearn.ensemble import RandomForestClassifier
class RandomForestClassification(DecisionTreeClassification):
    def fitToTrainingSet(self, **kwargs):
        self.classifier = RandomForestClassifier(**kwargs)
        classifier.fit(self.X_train, self.y_train)

