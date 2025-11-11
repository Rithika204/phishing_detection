<<<<<<< HEAD
import joblib
from tensorflow.keras.models import load_model
import os

# Check CNN model
cnn_path = "models/cnn_model.h5"
if os.path.exists(cnn_path):
    try:
        cnn = load_model(cnn_path)
        print("✅ CNN model loaded:", cnn)
    except Exception as e:
        print("❌ CNN load error:", e)
else:
    print("❌ CNN model file not found")

# Check XGBoost / placeholder model
xgb_path = "models/xgb_model.h5"
if os.path.exists(xgb_path):
    try:
        xgb_model = joblib.load(xgb_path)
        print("✅ XGBoost model loaded:", xgb_model)
    except Exception as e:
        print("❌ XGBoost load error:", e)
else:
    print("❌ XGBoost model file not found")
=======
import joblib
from tensorflow.keras.models import load_model
import os

# Check CNN model
cnn_path = "models/cnn_model.h5"
if os.path.exists(cnn_path):
    try:
        cnn = load_model(cnn_path)
        print("✅ CNN model loaded:", cnn)
    except Exception as e:
        print("❌ CNN load error:", e)
else:
    print("❌ CNN model file not found")

# Check XGBoost / placeholder model
xgb_path = "models/xgb_model.h5"
if os.path.exists(xgb_path):
    try:
        xgb_model = joblib.load(xgb_path)
        print("✅ XGBoost model loaded:", xgb_model)
    except Exception as e:
        print("❌ XGBoost load error:", e)
else:
    print("❌ XGBoost model file not found")
>>>>>>> c9dc7fe (Initial commit)
