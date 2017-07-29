#Date: 2017/07/29

from data_preprocessing import DataPreprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
class NLP(DataPreprocessing):
    def importDataset(self, file, delimiter, quoting):
        self.dataset = pd.read_csv(file, delimiter, quoting)
        
    def cleanTexts(self):
        #
        nltk.download('stopwords')
        self.corpus = []
        for i in range(0, 1000):
            review = re.sub('[^a-zA-Z]', ' ', self.dataset['Review'][i])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            self.corpus.append(review)

    def createBagOfWordsModel(self):
        #
        cv = CountVectorizer(max_features = 1500)
        self.X = cv.fit_transform(self.corpus).toarray()
        self.y = self.dataset.iloc[:, 1].values
    
    def fitNaiveBayesToTrainingSet(self):
        self.classifier = GaussianNB()
        self.classifier.fit(self.X_train, self.y_train)
        
    def predictResults(self):
        #
        self.y_pred = self.classifier.predict(self.X_test)
    
    def makeConfusionMatrix(self):
        self.cm = confusion_matrix(y_test, y_pred)
