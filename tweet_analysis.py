# Sentiment Analysis of Disaster tweets

# Importing the library
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

# Importing the dataset
dataset = pd.read_csv('train.csv')
ds = pd.read_csv('test.csv')
X1 = ds.iloc[:, -1].values

p = pd.DataFrame(dataset['keyword'].unique())
k = dataset[dataset['location'].isna()]

# Cleaning up the data
from nltk.stem import PorterStemmer
corpus = []
for i in range(len(dataset)):
    main_words = dataset['text'][i]
    main_words = main_words.split()
    ps = PorterStemmer()
    main_words = [ps.stem(word) for word in main_words if not word in set(stopwords.words('english'))]
    main_words = ' '.join(main_words)
    corpus.append(main_words)

cs = []
for j in range(len(X1)):
    m = X1[j]
    m = m.split()
    ps1 = PorterStemmer()
    m = [ps1.stem(word) for word in m if not word in set(stopwords.words('english'))]
    m = ' '.join(m)
    cs.append(m)


# Creating Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 15000)
X = cv.fit_transform(corpus).toarray()
X1 = cv.transform(cs).toarray()
y = dataset.iloc[:, -1].values

# Assigning dataset into training and testing models
X_train, y_train = X, y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting random forest classifier to the training data
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 450, criterion = 'entropy',random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)
y_test = pd.read_csv('predicted.csv')
np.savetxt('.csv', y_pred, delimiter = ',')

# Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy111 = accuracy_score(y_test, y_pred)
