import pandas as pd
import numpy as np
df = pd.read_csv("dark_patterns.csv")
features = df[['Comment', 'Where in website?']]
target_category = df['Pattern Category']
target_type = df['Pattern Type']

df['Pattern String'] = df['Pattern String'].replace(np.nan, 'Graphical')
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
text_features = tfidf_vectorizer.fit_transform(df['Pattern String'])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
column_transformer = ColumnTransformer(
    transformers=[
        ('comment_and_website', OneHotEncoder(sparse =False), ['Comment', 'Where in website?'])
    ],
    remainder='passthrough'
)
categorical_features = column_transformer.fit_transform(features)

X = pd.concat([pd.DataFrame(text_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out()),
               pd.DataFrame(categorical_features, columns=column_transformer.get_feature_names_out(['Comment', 'Where in website?'])).reset_index(drop=True)], axis=1)

from sklearn.preprocessing import LabelEncoder
label_category = LabelEncoder()
label_type = LabelEncoder()

y_category = label_category.fit_transform(target_category)
y_type = label_type.fit_transform(target_type)

from sklearn.model_selection import train_test_split
X_train, X_test, y_category_train, y_category_test, y_type_train, y_type_test = train_test_split(
    X, y_category, y_type, test_size=0.3, random_state=42
)

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000))
classifier = MultiOutputClassifier(RandomForestClassifier())
classifier.fit(X_train, pd.DataFrame({'category': y_category_train, 'type': y_type_train}))

y_category_pred, y_type_pred = zip(*classifier.predict(X_test))
from sklearn.metrics import accuracy_score
accuracy_category = accuracy_score(y_category_test, y_category_pred)
accuracy_type = accuracy_score(y_type_test, y_type_pred)

from joblib import dump
dump(classifier, 'dark_pattern_detector.joblib')
dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
dump(column_transformer, 'column_transformer.joblib')

