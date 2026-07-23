# 🫁 ChestMNIST-KNN-Baseline

A classical machine learning approach built from scratch to establish a robust baseline for medical image classification. Implementing core architectural concepts from the Stanford CS231n series, this repository applies a fully vectorized K-Nearest Neighbor (KNN) classifier and K-Fold cross-validation to chest radiography data.

The project highlights the technical hurdles of processing medical imagery using classical distance metrics, specifically focusing on data pipelines, memory optimization, and the **Accuracy Paradox** introduced by clinical class imbalance.

## 🎯 Objective & Clinical Scope
The primary goal is to evaluate how effectively classical, non-parametric spatial algorithms can identify structural pathologies in radiography without the benefit of a deep feature extractor (like a CNN).

* **Dataset:** ChestMNIST (MedMNIST v2) — pre-processed 28x28 digital matrices.
* **Target Pathology:** **Cardiomegaly** (Enlarged Heart), selected because it induces massive, macro-structural geometric alterations to the center of the pixel arrays.

## 🛠️ Key Technical Milestones

* **Zero-Loop Matrix Vectorization:** Rather than calculating distances element-by-element using highly inefficient nested loops, the L2 distance calculation is mathematically expanded into a fully vectorized dot product. This leverages high-performance linear algebra broadcasting, optimized to run complex distance tensors locally on unified memory architectures in seconds.
* **Type Casting & Memory Safety:** Fixed silent numerical bugs by casting raw `uint8` imagery to `float32` and scaling inputs to a `[0, 1]` distribution, preventing critical integer wrap-around overflows during matrix squaring.
* **Algorithmic Class Balancing:** Solved severe data skew (97% healthy patients) by implementing random majority class undersampling, establishing a perfect 50/50 distribution to force true pattern learning.
* **Hyperparameter Search:** Engineered a rigorous 5-Fold Cross-Validation search loop to optimize the neighbor voting parameters.

## 📊 Evaluation & The Accuracy Paradox

When dealing with highly imbalanced medical data, raw accuracy is a dangerous metric. 
* **The Trap:** An unbalanced, naive model easily achieved a **97.41% raw accuracy** score by taking advantage of class imbalance (blindly predicting "Healthy" for every target). However, a Confusion Matrix revealed it yielded **0 True Positives**, rendering it clinically useless.
* **The Solution:** By enforcing strict class balancing, the optimized baseline model shifted performance dramatically. While raw accuracy dropped, the **True Positive Disease Recall jumped to ~63%**. 

**Key Algorithmic Insight:** While the model successfully learns structural cues to identify the pathology, flattening a 2D chest X-ray array into a 1D vector completely breaks spatial context. The Euclidean metric becomes highly sensitive to non-clinical background noise, leading to an elevated False Positive rate. This explicitly defines the performance ceiling for classical ML and demonstrates the strict necessity of progressing to Convolutional Neural Networks (CNNs).

## 🚀 Running the Interactive Dashboard

This repository includes a Streamlit web application to visualize the Confusion Matrix and adjust hyperparameters in real-time.

**1. Clone the repository and navigate to the directory:**
```bash
git clone [https://github.com/Tendor517/Classifying-Cardiomegaly-The-KNN-Baseline.git](https://github.com/Tendor517/Classifying-Cardiomegaly-The-KNN-Baseline.git)
cd Classifying-Cardiomegaly-The-KNN-Baseline

## 🛠️ Developer Guide: Local Setup & Execution Pipeline

After cloning the repository, install the dependencies, and execute the machine learning pipeline from scratch, follow the step-by-step instructions below.

### 1. Install Required Packages
Open your terminal inside the project directory and run the following command to install all necessary Python packages for data processing, classical ML, visualization, and web deployment:
```bash
pip install numpy scikit-learn matplotlib seaborn medmnist streamlit

Step 1: Understand the Core Engine (knn.py)
What it is: This is the standalone utility module containing the custom KNearestNeighbor class and the zero-loop vectorized L2 distance calculation.
Action Required: Do not run this file directly. It acts as the backend engine and is automatically imported by the execution scripts below.

Step 2: Run Hyperparameter Optimization (kfold_cross_validation_loop.py)
Purpose: Executes a 5-Fold Cross-Validation search across various k-values (k∈[1,20]) to discover the optimal number of neighbors for cardiomegaly detection.
```bash
python3 kfold_cross_validation_loop.py

Step 3: Execute Full Model Evaluation (balanced_training.py)
Purpose: Runs the end-to-end production pipeline. It loads the ChestMNIST dataset, applies majority-class undersampling to fix the 97% data skew, normalizes image arrays to [0,1], and evaluates the optimal model.
```bash
python3 balanced_training.py

Step 4: Launch the Interactive Web Dashboard (app.py)
Purpose: Starts a local web server hosting the frontend Streamlit UI, allowing you or end-users to dynamically adjust k-values and inspect confusion matrix heatmaps in real time.
```bash
streamlit run app.py