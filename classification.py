"""
Name: Jessica Kam
Date: 2017/07/28
"""

from data_preprocessing import DataPreprocessing
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


class Classification(DataPreprocessing):
    def __init__(self):
        pass #
        
    def importDataset(self, csv_file, list_of_columns):
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
   
    def visualizeResults(self, color1, color2, what_visualizing, xlabel, ylabel):
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, self.classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap((color1, color2)))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap((color1, color2))(i), label = j)
        plt.title(self.generateTitle(what_visualizing))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
    
    def generateTitle(self, what_visualizing):
        return self.name + ' ' + '(' + what_visualizing + ')'
    
    def visualizeTrainingSetResults(self, color1, color2, xlabel, ylabel):
        self.visualizeResults(color1, color2, what_visualizing='Training Set', xlabel, ylabel)
     
    def visualizeTestSetResults(self, color1, color2, xlabel, ylabel):
        self.visualizeResults(=color1, color2, what_visualizing='Test Set', xlabel, ylabel)

from sklearn.linear_model import LogisticRegression
class LogisticRegression(Classification):
    def __init__(self):
        self.name = 'LogisticRegression'
        
    def fitToTrainingSet(self, **kwargs):
        self.classifier = LogisticRegression(**kwargs)
        self.classifier.fit(self.X_train, self.y_train)
    
    def predictTestResults(self):
        self.y_pred = self.classifier.predict(self.X_test)

form sklearn.neighbors import KNeighborsClassifier
class KNN(Classification):
    def __init__(self):
        self.name = 'KNN'
        
    def fitToTrainingSet(self, **kwargs):
        self.classifier = KNeighborsClassifier(**kwargs)
        self.classifier.fit(self.X_train, self.y_train)
        
from sklearn.svm import SVC
class SVM(Classification):
    def __init__(self):
        self.name = 'SVM'
        
    def fitToTrainingSet(self, **kwargs):
        self.classifier = SVC(**kwargs)
        self.classifier.fit(self.X_train, self.y_train)


class KernelSVM(SVM):
    def __init__(self):
        self.name = 'KernelSVM'
        
    def fitToTrainingSet(self, **kwargs):
        super(KernelSVM, self).fitToTrainingSet(kernel='rbf', **kwargs)

from sklearn.naive_bayes import GaussianNB
class NaiveBayes(Classification):
    def __init__(self):
        self.name = 'NaiveBayes'
        
    def fitToTrainingSet(self, **kwargs):
        self.classifer = GaussianNB()
        self.classifer.fit(self.X_train, self.y_train)

from sklearn.tree import DecisionTreeClassifier
class DecisionTreeClassification(Classification):
    def __init__(self):
        self.name = 'DecisonTreeClassif'
    
    def fitToTrainingSet(self, **kwargs):
        self.classifier = DecisionTreeClassifier(**kwargs)
        self.classifier.fit(self.X_train, self.y_train)

from sklearn.ensemble import RandomForestClassifier
class RandomForestClassification(DecisionTreeClassification):
    def __init__(self);:
        self.name = 'RandomForestClassif'
        
    def fitToTrainingSet(self, **kwargs):
        self.classifier = RandomForestClassifier(**kwargs)
        classifier.fit(self.X_train, self.y_train)
