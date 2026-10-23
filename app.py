from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# AI model load karo
with open("sentiment_model.pkl", "rb") as file:
    vectorizer, model = pickle.load(file)


@app.route("/")
def home():
  return render_template("index.html")
@app.route("/dashboard")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")
@app.route("/analytics")
def analytics():
    return render_template("analytics.html")
@app.route("/business")
def business():
    return render_template("business.html")
def dashboard():
    return render_template("dashboard.html")
@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    review = data["review"]

    review_vectorized = vectorizer.transform([review])

    prediction = model.predict(review_vectorized)[0]

    return jsonify({
        "review": review,
        "sentiment": prediction
    })


if __name__ == "__main__":
    app.run(debug=True)