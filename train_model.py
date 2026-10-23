import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Dataset load karo
data = pd.read_csv("dataset/reviews.csv")
# Review aur sentiment alag karo
X = data["review"]
y = data["sentiment"]

# Text ko numerical features mein convert karo
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Machine Learning model
model = LogisticRegression()

# Model train karo
model.fit(X_vectorized, y)

# Model aur vectorizer save karo
with open("sentiment_model.pkl", "wb") as file:
    pickle.dump((vectorizer, model), file)

print("Model training complete!")