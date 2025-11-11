<<<<<<< HEAD
# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

from utils import extract_basic_features

# Load dataset
df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
df = df.dropna(subset=['URL'])
df['label'] = df['label'].astype(int)

X_urls = df['URL']
y = df['label']

# Split data
X_train_urls, X_test_urls, y_train, y_test = train_test_split(X_urls, y, test_size=0.2, random_state=42, stratify=y)

# --- Tokenizer for CNN ---
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train_urls)
X_train_seq = tokenizer.texts_to_sequences(X_train_urls)
X_test_seq = tokenizer.texts_to_sequences(X_test_urls)

X_train_seq = pad_sequences(X_train_seq, maxlen=100)
X_test_seq = pad_sequences(X_test_seq, maxlen=100)

vocab_size = len(tokenizer.word_index) + 1

# --- CNN Model ---
cnn = Sequential([
    Embedding(vocab_size, 64, input_length=100),
    Conv1D(64, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_seq, y_train, epochs=3, batch_size=256, validation_split=0.1)

# Save CNN
cnn.save("models/cnn_model.h5")

# --- CNN Confusion Matrix ---
y_pred_cnn = (cnn.predict(X_test_seq) > 0.5).astype(int)
cm_cnn = confusion_matrix(y_test, y_pred_cnn)
print("\n🔹 CNN Confusion Matrix:")
print(cm_cnn)
print("\nClassification Report (CNN):")
print(classification_report(y_test, y_pred_cnn))

# --- XGBoost Model ---
X_train_features = np.array([extract_basic_features(u)[0] for u in X_train_urls])
X_test_features = np.array([extract_basic_features(u)[0] for u in X_test_urls])

xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
xgb.fit(X_train_features, y_train)
xgb.save_model("models/xgb_model.json")

# --- XGBoost Confusion Matrix ---
y_pred_xgb = xgb.predict(X_test_features)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print("\n🔹 XGBoost Confusion Matrix:")
print(cm_xgb)
print("\nClassification Report (XGBoost):")
print(classification_report(y_test, y_pred_xgb))

# Save tokenizer
joblib.dump(tokenizer, "models/tokenizer.pkl")

print("\n✅ Models trained, evaluated, and saved successfully!")
=======
# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

from utils import extract_basic_features

# Load dataset
df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
df = df.dropna(subset=['URL'])
df['label'] = df['label'].astype(int)

X_urls = df['URL']
y = df['label']

# Split data
X_train_urls, X_test_urls, y_train, y_test = train_test_split(X_urls, y, test_size=0.2, random_state=42, stratify=y)

# --- Tokenizer for CNN ---
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train_urls)
X_train_seq = tokenizer.texts_to_sequences(X_train_urls)
X_test_seq = tokenizer.texts_to_sequences(X_test_urls)

X_train_seq = pad_sequences(X_train_seq, maxlen=100)
X_test_seq = pad_sequences(X_test_seq, maxlen=100)

vocab_size = len(tokenizer.word_index) + 1

# --- CNN Model ---
cnn = Sequential([
    Embedding(vocab_size, 64, input_length=100),
    Conv1D(64, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_seq, y_train, epochs=3, batch_size=256, validation_split=0.1)

# Save CNN
cnn.save("models/cnn_model.h5")

# --- CNN Confusion Matrix ---
y_pred_cnn = (cnn.predict(X_test_seq) > 0.5).astype(int)
cm_cnn = confusion_matrix(y_test, y_pred_cnn)
print("\n🔹 CNN Confusion Matrix:")
print(cm_cnn)
print("\nClassification Report (CNN):")
print(classification_report(y_test, y_pred_cnn))

# --- XGBoost Model ---
X_train_features = np.array([extract_basic_features(u)[0] for u in X_train_urls])
X_test_features = np.array([extract_basic_features(u)[0] for u in X_test_urls])

xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
xgb.fit(X_train_features, y_train)
xgb.save_model("models/xgb_model.json")

# --- XGBoost Confusion Matrix ---
y_pred_xgb = xgb.predict(X_test_features)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print("\n🔹 XGBoost Confusion Matrix:")
print(cm_xgb)
print("\nClassification Report (XGBoost):")
print(classification_report(y_test, y_pred_xgb))

# Save tokenizer
joblib.dump(tokenizer, "models/tokenizer.pkl")

print("\n✅ Models trained, evaluated, and saved successfully!")
>>>>>>> c9dc7fe (Initial commit)
