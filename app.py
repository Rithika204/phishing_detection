from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from utils import extract_basic_features

app = Flask(__name__)
CORS(app)

# Load XGBoost model only
xgb_model = joblib.load("models/xgb_model.h5")

@app.route("/")
def home():
    return render_template("phish.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"})

    try:
        xgb_input = extract_basic_features(url)
        pred = xgb_model.predict_proba(xgb_input)[0][1]

        prediction = "phishing" if pred > 0.5 else "legitimate"

        return jsonify({
            "prediction": prediction,
            "reason": f"XGBoost Score: {pred:.2f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
