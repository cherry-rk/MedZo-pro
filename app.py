from flask import Flask, request, render_template, jsonify
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer from model folder
MODEL_PATH = os.path.join("model", "model.pkl")

VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")


try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print("Error loading model or vectorizer:", e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("message")
    if not user_input:
        return jsonify({"response": "Please enter some symptoms."})

    try:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        return jsonify({"response": f"Based on your symptoms, you may have: {prediction[0]}."})
    except Exception as e:
        return jsonify({"response": f"Error during prediction: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

