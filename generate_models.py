<<<<<<< HEAD
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Make models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# --- Create a tiny CNN model ---
model = Sequential([
    Flatten(input_shape=(8,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train briefly on random data just to save a valid model
x = np.random.rand(100, 8)
y = (np.sum(x, axis=1) > 4).astype(int)
model.fit(x, y, epochs=1, verbose=0)

# Save CNN model
model.save('models/cnn_model.h5')

# --- Create a placeholder XGBoost model ---
from sklearn.ensemble import RandomForestClassifier
import joblib

X = np.random.rand(100, 8)
Y = (np.sum(X, axis=1) > 4).astype(int)
clf = RandomForestClassifier(n_estimators=5)
clf.fit(X, Y)

joblib.dump(clf, 'models/xgb_model.h5')

print("✅ Models created successfully in the 'models' folder!")
=======
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Make models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# --- Create a tiny CNN model ---
model = Sequential([
    Flatten(input_shape=(8,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train briefly on random data just to save a valid model
x = np.random.rand(100, 8)
y = (np.sum(x, axis=1) > 4).astype(int)
model.fit(x, y, epochs=1, verbose=0)

# Save CNN model
model.save('models/cnn_model.h5')

# --- Create a placeholder XGBoost model ---
from sklearn.ensemble import RandomForestClassifier
import joblib

X = np.random.rand(100, 8)
Y = (np.sum(X, axis=1) > 4).astype(int)
clf = RandomForestClassifier(n_estimators=5)
clf.fit(X, Y)

joblib.dump(clf, 'models/xgb_model.h5')

print("✅ Models created successfully in the 'models' folder!")
>>>>>>> c9dc7fe (Initial commit)
