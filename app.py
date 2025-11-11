import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

# Only import TF/XGB when needed to avoid heavy startup
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

from utils import extract_basic_features, preprocess_for_cnn

app = Flask(__name__)
CORS(app)

# Lazy globals
_cnn = None
_xgb = None
_tokenizer = None

def load_artifacts():
    global _cnn, _xgb, _tokenizer
    if _cnn is None:
        _cnn = load_model("models/cnn_model.h5")               # exists in your repo
    if _xgb is None:
        # use the .pkl you have (NOT .h5)
        _xgb = joblib.load("models/xgboost_model.pkl")
        if isinstance(_xgb, XGBClassifier) is False:
            # in case the pickle wraps the estimator
            try:
                _xgb = _xgb["model"]
            except Exception:
                pass
    if _tokenizer is None:
        _tokenizer = joblib.load("models/tokenizer.pkl")

@app.route("/")
def home():
    return render_template("phish.html")  # file is in templates/

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    load_artifacts()

    # CNN
    cnn_input = preprocess_for_cnn(url, _tokenizer)
    cnn_score = float(_cnn.predict(cnn_input, verbose=0)[0][0])

    # XGBoost
    xgb_input = extract_basic_features(url)
    xgb_score = float(_xgb.predict_proba(xgb_input)[0][1])

    final = (cnn_score + xgb_score) / 2.0
    label = "phishing" if final > 0.5 else "legitimate"

    return jsonify({
        "prediction": label,
        "reason": f"CNN: {cnn_score:.2f}, XGB: {xgb_score:.2f}, Final: {final:.2f}"
    })
