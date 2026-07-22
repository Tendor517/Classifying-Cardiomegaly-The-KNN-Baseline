import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from knn import KNearestNeighbor

# 1. Load Data
print("Loading dataset...")
data = np.load("chestmnist.npz")

# --- THE FIX: Cast to float and normalize to [0, 1] ---
X_train_raw = data["train_images"].reshape(data["train_images"].shape[0], -1).astype(float) / 255.0
y_train_raw = data["train_labels"][:, 1]

X_test = data["test_images"].reshape(data["test_images"].shape[0], -1).astype(float) / 255.0
y_test = data["test_labels"][:, 1]
# ------------------------------------------------------

# 2. Balance the Training Data (Undersampling the negatives)
print("\nBalancing the training set...")
pos_indices = np.where(y_train_raw == 1)[0]
neg_indices = np.where(y_train_raw == 0)[0]

# Randomly sample the exact same number of negatives as we have positives
np.random.seed(42) # For reproducible results
sampled_neg_indices = np.random.choice(neg_indices, len(pos_indices), replace=False)

# Combine and shuffle so the algorithm doesn't memorize the order
balanced_indices = np.concatenate([pos_indices, sampled_neg_indices])
np.random.shuffle(balanced_indices)

X_train_balanced = X_train_raw[balanced_indices]
y_train_balanced = y_train_raw[balanced_indices]

print(f"Original Training Shape: {X_train_raw.shape}")
print(f"New Balanced Training Shape: {X_train_balanced.shape}")
print(f"Total Positives: {np.sum(y_train_balanced)} | Total Negatives: {len(y_train_balanced) - np.sum(y_train_balanced)}")

# 3. Train and Predict
print("\nTraining model on balanced dataset...")
classifier = KNearestNeighbor()
classifier.train(X_train_balanced, y_train_balanced)

# We use k=5 here. Since the dataset is now balanced and smaller, 
# looking at slightly more neighbors helps smooth out the noise.
print("Predicting on the untouched Test Set (this will be fast)...")
y_test_pred = classifier.predict(X_test, k=5)

# 4. The Final Reveal
print("\n--- Balanced Model Evaluation ---")
acc = np.mean(y_test_pred == y_test)
print(f"Raw Test Accuracy: {acc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Negative", "Positive"]))