import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


df = pd.read_csv("normie.csv")

df = df.rename(columns={'classification': 'Label'})
df = df.rename(columns={'Pattern String': 'Pattern'})

df.dropna(inplace=True)


X_train, X_test, Y_train, y_test = train_test_split(df['Pattern'], df['Label'], test_size = 0.3, random_state = 0)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

RFC =  RandomForestClassifier()
RFC.fit(X_train_vectorized, Y_train)
y_pred = RFC.predict(X_test_vectorized)

dump(RFC, 'presence_classifer.joblib')
dump(vectorizer, 'presence_vectorizer.joblib')
