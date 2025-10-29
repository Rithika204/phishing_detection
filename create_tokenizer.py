import joblib
from tensorflow.keras.preprocessing.text import Tokenizer

# Create a simple placeholder tokenizer
tokenizer = Tokenizer(num_words=100)
# Fit on some dummy text so it’s valid
dummy_texts = ["http example site test phishing safe"]
tokenizer.fit_on_texts(dummy_texts)

# Save to models folder
import os
os.makedirs("models", exist_ok=True)
joblib.dump(tokenizer, "models/tokenizer.pkl")

print("✅ Placeholder tokenizer.pkl created in models/")
