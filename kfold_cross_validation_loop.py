# filename: kfold_cross_validation_loop.py
import numpy as np
import matplotlib.pyplot as plt
from knn import KNearestNeighbor # Assuming your class is in knn.py

# 1. Loading the dataset
print("Loading the dataset...")
data = np.load("chestmnist.npz")
X_train_raw = data["train_images"]
y_train_raw = data["train_labels"]

X_test_raw = data["test_images"]
y_test_raw = data["test_labels"]

# 2. Flatten the 28x28 images into 784D vectors for your KNN
# X_train shape becomes (112180, 784)
X_train = X_train_raw.reshape(X_train_raw.shape[0], -1) 
X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)

# 3. Isolate the Cardiomegaly label (Index 1)
# y_train shape becomes (112180,)
CARDIOMEGALY_INDEX = 1
y_train = y_train_raw[:, CARDIOMEGALY_INDEX]
y_test = y_test_raw[:, CARDIOMEGALY_INDEX]

print(f"Original Training data shape: {X_train.shape}")

# --- THE FIX: SUBSET THE DATA ---
# Downsample to 10,000 images to prevent out-of-memory errors
# during the vectorized distance matrix calculation.
num_cv_training = 10000 
X_train = X_train[:num_cv_training]
y_train = y_train[:num_cv_training]

print(f"Subsampled training data shape for CV: {X_train.shape}")
print(f"Subsampled positive Cardiomegaly cases: {np.sum(y_train)}")
# ---------------------------------

# 4. Define hyperparameter search space
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20]

# 5. Split the data into folds
print(f"Splitting data into {num_folds} folds...")
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

# Dictionary to hold the accuracies for each k
k_to_accuracies = {k: [] for k in k_choices}

# 6. Execute Cross-Validation
print("Starting K-Fold Cross-Validation...")
for k in k_choices:
    for i in range(num_folds):
        # Isolate the current fold for validation
        X_val_fold = X_train_folds[i]
        y_val_fold = y_train_folds[i]
        
        # Concatenate the remaining folds for training
        X_train_fold = np.concatenate([X_train_folds[j] for j in range(num_folds) if j != i])
        y_train_fold = np.concatenate([y_train_folds[j] for j in range(num_folds) if j != i])
        
        # Initialize and train the model
        classifier = KNearestNeighbor()
        classifier.train(X_train_fold, y_train_fold)
        
        # Predict on the validation fold
        y_val_pred = classifier.predict(X_val_fold, k=k)
        
        # Calculate accuracy
        num_correct = np.sum(y_val_pred == y_val_fold)
        accuracy = float(num_correct) / len(y_val_fold)
        
        # Store the result
        k_to_accuracies[k].append(accuracy)
        
    print(f"Evaluated k = {k} | Mean Accuracy: {np.mean(k_to_accuracies[k]):.4f}")

# 7. Plot the cross-validation results
print("Generating plot...")
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies, color='blue', alpha=0.5)

# Plot the trend line with error bars
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])

plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std, color='black', marker='o')
plt.title('Cross-Validation on k (ChestMNIST - Cardiomegaly)')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.xticks(k_choices)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 8. Extract the absolute best k
best_k = k_choices[np.argmax(accuracies_mean)]
print(f"\nOptimal value for k based on cross-validation: {best_k}")