# train.py
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os

# Paths
csv_path = "data/waste_labels.csv"   # your CSV
img_base_dir = "data"                # folder containing organic/ and recyclable/
img_size = (128, 128)
batch_size = 32

# Load CSV
df = pd.read_csv(csv_path)
classes = sorted(df['label'].unique())
print("Classes:", classes)

# Mapping labels to integers
label_to_idx = {label: idx for idx, label in enumerate(classes)}

# Load images
X = []
y = []

for _, row in df.iterrows():
    # Prepend folder name based on label
    folder = row['label']  # 'recyclable' or 'organic'
    img_path = os.path.join(img_base_dir, folder, row['filename'])
    img_path = os.path.normpath(img_path)
    
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)/255.0  # normalize
        X.append(img_array)
        y.append(label_to_idx[row['label']])
    else:
        print("Missing file:", img_path)

X = np.array(X)
y = np.array(y)
print(f"Total images loaded: {len(X)}")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(classes), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=batch_size
)

# Evaluate on validation data
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n✅ Validation Accuracy: {val_acc * 100:.2f}%")

# Save model + classes
model.save("waste_cnn.h5")
with open("classes.txt", "w") as f:
    f.write("\n".join(classes))
