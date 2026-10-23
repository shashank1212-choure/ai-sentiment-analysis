import pickle

with open("sentiment_model.pkl", "rb") as file:
    vectorizer, model = pickle.load(file)

review = input("Enter your review: ")

review_vectorized = vectorizer.transform([review])

prediction = model.predict(review_vectorized)[0]

if prediction == "positive":
    print("😊 POSITIVE")
elif prediction == "negative":
    print("😡 NEGATIVE")
else:
    print("😐 NEUTRAL")