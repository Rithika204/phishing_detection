# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import joblib
import traceback

app = Flask(__name__, template_folder="templates")
CORS(app)

# lazy-loaded globals
cnn_model = None
xgb_model = None
tokenizer = None

def load_models():
    global cnn_model, xgb_model, tokenizer
    if cnn_model is not None and xgb_model is not None and tokenizer is not None:
        return

    # Use try/except so the server doesn't crash on startup
    try:
        # For keras model
        from tensorflow.keras.models import load_model as keras_load_model
        if os.path.exists("models/cnn_model.h5"):
            cnn_model = keras_load_model("models/cnn_model.h5")
        else:
            print("Warning: models/cnn_model.h5 not found")

        # XGBoost model saved with joblib
        if os.path.exists("models/xgb_model.h5"):
            xgb_model = joblib.load("models/xgb_model.h5")
        elif os.path.exists("models/xgboost_model.pkl"):
            xgb_model = joblib.load("models/xgboost_model.pkl")
        else:
            print("Warning: XGBoost model not found")

        # tokenizer
        if os.path.exists("models/tokenizer.pkl"):
            tokenizer = joblib.load("models/tokenizer.pkl")
        else:
            print("Warning: tokenizer.pkl not found")

    except Exception as e:
        print("Error loading models:", e)
        traceback.print_exc()

# import feature functions only after load to keep module imports light
from utils import extract_basic_features, preprocess_for_cnn

@app.route("/")
def home():
    return render_template("phish.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_models()
        data = request.get_json(force=True)
        url = data.get("url", "")
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        # CNN
        cnn_pred = 0.0
        if cnn_model is not None and tokenizer is not None:
            cnn_input = preprocess_for_cnn(url, tokenizer)
            cnn_pred = float(cnn_model.predict(cnn_input)[0][0])

        # XGBoost
        xgb_pred = 0.0
        if xgb_model is not None:
            xgb_input = extract_basic_features(url)
            try:
                xgb_pred = float(xgb_model.predict_proba(xgb_input)[0][1])
            except Exception:
                xgb_pred = float(xgb_model.predict(xgb_input)[0])

        # Combine
        final_score = (cnn_pred + xgb_pred) / ( (1 if cnn_model is None else 1) + (1 if xgb_model is None else 1) ) * 2 if (cnn_model is None or xgb_model is None) else (cnn_pred + xgb_pred) / 2
        prediction = "phishing" if final_score > 0.5 else "legitimate"

        return jsonify({
            "prediction": prediction,
            "reason": f"CNN: {cnn_pred:.2f}, XGBoost: {xgb_pred:.2f}, Final: {final_score:.2f}"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Optional local run (useful for testing)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
