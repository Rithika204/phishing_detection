from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
from tensorflow.keras.models import load_model
from utils import extract_basic_features, preprocess_for_cnn

app = Flask(__name__)
CORS(app)

# Load trained models
cnn_model = load_model("models/cnn_model.h5")
xgb_model = joblib.load("models/xgboost_model.pkl")
tokenizer = joblib.load("models/tokenizer.pkl")

@app.route("/")
def home():
    return render_template("phish.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"})

    # CNN Prediction
    cnn_input = preprocess_for_cnn(url, tokenizer)
    cnn_pred = cnn_model.predict(cnn_input)[0][0]

    # XGBoost Prediction
    xgb_input = extract_basic_features(url)
    xgb_pred = xgb_model.predict_proba(xgb_input)[0][1]

    # Combine predictions
    final_score = (cnn_pred + xgb_pred) / 2
    prediction = "phishing" if final_score > 0.5 else "legitimate"

    return jsonify({
        "prediction": prediction,
        "reason": f"CNN: {cnn_pred:.2f}, XGBoost: {xgb_pred:.2f}, Final: {final_score:.2f}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
