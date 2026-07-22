import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from knn import KNearestNeighbor # Your custom class

# 1. Load Data
print("Loading dataset...")
data = np.load("chestmnist.npz")

X_train = data["train_images"].reshape(data["train_images"].shape[0], -1).astype(float) / 255.0
y_train = data["train_labels"][:, 1] # Index 1 for Cardiomegaly

X_test = data["test_images"].reshape(data["test_images"].shape[0], -1).astype(float) / 255.0
y_test = data["test_labels"][:, 1]

# 2. Subset Training Data (to match our CV environment)
X_train = X_train[:10000]
y_train = y_train[:10000]

# 3. Train final model and Predict on TEST set
print("Training final model and predicting on Test Set...")
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# We use k=1 as outputted by your cross-validation
y_test_pred = classifier.predict(X_test, k=1)

# 4. The Reveal: Confusion Matrix
print("\n--- Final Evaluation ---")
acc = np.mean(y_test_pred == y_test)
print(f"Raw Test Accuracy: {acc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Negative", "Positive"]))