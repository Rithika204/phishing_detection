<<<<<<< HEAD
# test_hybrid_model.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from utils import extract_basic_features

# --- Load dataset ---
df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
df = df.dropna(subset=['URL'])
df['label'] = df['label'].astype(int)
X_urls = df['URL']
y = df['label']

# --- Load saved models and tokenizer ---
cnn = load_model("models/cnn_model.h5")
xgb = XGBClassifier()
xgb.load_model("models/xgb_model.json")
tokenizer = joblib.load("models/tokenizer.pkl")

# --- Prepare CNN input ---
X_seq = tokenizer.texts_to_sequences(X_urls)
X_seq = pad_sequences(X_seq, maxlen=100)

# --- Prepare XGBoost input ---
X_features = np.array([extract_basic_features(u)[0] for u in X_urls])

# --- Predictions ---
cnn_preds = (cnn.predict(X_seq) > 0.5).astype(int).flatten()
xgb_preds = xgb.predict(X_features)

# --- Combine for Hybrid (average voting) ---
hybrid_preds = np.round((cnn_preds + xgb_preds) / 2)

# --- Evaluation ---
cm = confusion_matrix(y, hybrid_preds)
acc = accuracy_score(y, hybrid_preds)
print("✅ Hybrid Model Accuracy:", acc)
print("\nHybrid Confusion Matrix:\n", cm)
print("\nClassification Report (Hybrid):\n", classification_report(y, hybrid_preds))

# --- Plot Confusion Matrix ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title(f"Hybrid Model Confusion Matrix\nAccuracy: {acc:.4f}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
=======
# test_hybrid_model.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from utils import extract_basic_features

# --- Load dataset ---
df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
df = df.dropna(subset=['URL'])
df['label'] = df['label'].astype(int)
X_urls = df['URL']
y = df['label']

# --- Load saved models and tokenizer ---
cnn = load_model("models/cnn_model.h5")
xgb = XGBClassifier()
xgb.load_model("models/xgb_model.json")
tokenizer = joblib.load("models/tokenizer.pkl")

# --- Prepare CNN input ---
X_seq = tokenizer.texts_to_sequences(X_urls)
X_seq = pad_sequences(X_seq, maxlen=100)

# --- Prepare XGBoost input ---
X_features = np.array([extract_basic_features(u)[0] for u in X_urls])

# --- Predictions ---
cnn_preds = (cnn.predict(X_seq) > 0.5).astype(int).flatten()
xgb_preds = xgb.predict(X_features)

# --- Combine for Hybrid (average voting) ---
hybrid_preds = np.round((cnn_preds + xgb_preds) / 2)

# --- Evaluation ---
cm = confusion_matrix(y, hybrid_preds)
acc = accuracy_score(y, hybrid_preds)
print("✅ Hybrid Model Accuracy:", acc)
print("\nHybrid Confusion Matrix:\n", cm)
print("\nClassification Report (Hybrid):\n", classification_report(y, hybrid_preds))

# --- Plot Confusion Matrix ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title(f"Hybrid Model Confusion Matrix\nAccuracy: {acc:.4f}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
>>>>>>> c9dc7fe (Initial commit)
