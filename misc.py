"""
unit9
dimensionality reduction
PCA
LDA
kernel PCA

unit10
GridSearch
KFoldCrossValidation
Model Selection
XG Boost

#Date: 2017/07/29
"""

from data_preprocessing import DataPreprocessing

class DimensionalityReduction(DataPreprocessing):
    def __init__(self):
        self.name = ''
        self.abbrev = ''
        
    def scaleFeatures(self):
        #
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
    def fitLogisticRegToDataset(self):
        #
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
        
    def predictResults(self):
        #
        y_pred = classifier.predict(X_test)
    
    def makeConfusionMatrix(self):
        #
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
    def generateTitle(self, what_visualizing):
        return 'Logistic Regression' + ' ' + '(' + what_visualizing + ')'
    
    def visualizeResults(self, what_visualizing, tuple_colors, xlabel, ylabel):
        from matplotlib.colors import ListedColormap
        X_set, y_set = self.X_train, self.y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(tuple_colors))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(tuple_colors)(i), label = j)
        plt.title(self.generateTitle(what_visualizing)) #'Logistic Regression (Training set)')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
        
    def visualizeTrainingSetResults(self, tuple_colors, xlabel, ylabel):
        self.visualizeResults(what_visualizing='Training Set', tuple_colors, xlabel, ylabel)
        
    def visualizeTestSetResults(self, tuple_colors, xlabel, ylabel):
        self.visualizeResults(what_visualizing='Test Set', tuple_colors, xlabel, ylabel)

class PCA(DimensionalityReduction):
    def __init__(self):
        self.name = 'PCA'
        
    def applyPCA(self):
        #
        from sklearn.decomposition import PCA
        pca = PCA(n_components = 2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        explained_variance = pca.explained_variance_ratio_
        

class LDA(DimensionalityReduction):
        def __init__(self):
        self.name = 'LDA'
        
    def applyLDA(self):
        #
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components = 2)
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)

class KernelPCA(PCA):
    def __init__(self):
        self.name = 'KernelPCA'
    
    def applyKernelPCA(self):
        #
        from sklearn.decomposition import KernelPCA
        kpca = KernelPCA(n_components = 2, kernel = 'rbf')
        X_train = kpca.fit_transform(X_train)
        X_test = kpca.transform(X_test)
####
class ModelSelection(DataPreprocessing):
    def scaleFeatures(self):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    
    def fitToTrainingSet(self, **kwargs):
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train)
    
    def predictResults(self):
        y_pred = self.classifier.predict(self.X_test)

    def makeConfusionMatrix(self):
        #
        self.cm = confusion_matrix(y_test, y_pred)

    def applyKFoldCrossValidation(estimator=self.classifier, X=self.X_train, y=self.y_train, cv=10):
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        accuracies.mean()
        accuracies.std()
        
    def applyGridSearchToFindBestModels(self):
        from sklearn.model_selection import GridSearchCV
        parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                      {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
        grid_search = GridSearchCV(estimator = classifier,
                                   param_grid = parameters,
                                   scoring = 'accuracy',
                                   cv = 10,
                                   n_jobs = -1)
        grid_search = grid_search.fit(X_train, y_train)
        best_accuracy = grid_search.best_score_
        best_parameters = grid_search.best_params_
        
    def visualizeTrainingSetResults(self):
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Kernel SVM (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()
    
    def visualizeTestSetResults(self):
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Kernel SVM (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
class XGBoost(DataPreprocessing):
    def fitToTrainingSet(self, **kwargs):
        #
        self.classifier = XGBClassifier()
        self.classifier.fit(self.X_train, self.y_train)
    
    def predictResults(self):
        #
        y_pred = self.classifier.predict(self.X_test)
        
    def makeConfusionMatrix(self):
        #
        self.cm = confusion_matrix(y_test, y_pred)

    def applyKFoldCrossValidation(self, **kwargs):
        ##
        self.accuracies = cross_val_score(**kwargs)
        self.accuracies.mean()
        self.accuracies.std()

#####


class SOM(DataPreprocessing):
    pass

class BoltzmannMachines(DataPreprocessing):
    pass

class AutoEncoder(DataPreprocessing):
    pass
