import streamlit as st
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from knn import KNearestNeighbor

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Page Setup
st.set_page_config(page_title="ChestMNIST KNN", layout="centered")
st.title("🫁 Chest X-Ray Classification Baseline")
st.markdown("A classical machine learning approach (K-Nearest Neighbors) to detecting Cardiomegaly from 28x28 medical arrays.")

# 2. Data Loading (Cached so it only runs once and doesn't freeze the app)
@st.cache_data
def load_and_balance_data():
    data = np.load("chestmnist.npz")
    
    # Load, flatten, and normalize
    X_train_raw = data["train_images"].reshape(data["train_images"].shape[0], -1).astype(float) / 255.0
    y_train_raw = data["train_labels"][:, 1]
    
    X_test = data["test_images"].reshape(data["test_images"].shape[0], -1).astype(float) / 255.0
    y_test = data["test_labels"][:, 1]
    
    # Balance the training set
    pos_indices = np.where(y_train_raw == 1)[0]
    neg_indices = np.where(y_train_raw == 0)[0]
    np.random.seed(42)
    sampled_neg_indices = np.random.choice(neg_indices, len(pos_indices), replace=False)
    balanced_indices = np.concatenate([pos_indices, sampled_neg_indices])
    np.random.shuffle(balanced_indices)
    
    X_train = X_train_raw[balanced_indices]
    y_train = y_train_raw[balanced_indices]
    
    return X_train, y_train, X_test, y_test

with st.spinner("Loading and balancing medical dataset..."):
    X_train, y_train, X_test, y_test = load_and_balance_data()

# 3. User Controls
st.sidebar.header("Model Hyperparameters")
# Creates a slider in the UI for the professor to change 'k'
k_value = st.sidebar.slider("Number of Neighbors (k)", min_value=1, max_value=25, value=5, step=2)

# 4. Model Execution
if st.button("Run Evaluation"):
    with st.spinner(f"Computing vectorized L2 distances for k={k_value}..."):
        classifier = KNearestNeighbor()
        classifier.train(X_train, y_train)
        y_test_pred = classifier.predict(X_test, k=k_value)
        
        # Metrics
        acc = np.mean(y_test_pred == y_test)
        cm = confusion_matrix(y_test, y_test_pred)
        

        # 5. UI Display
        st.success("Evaluation Complete!")
        
        # --- Upgraded Top Metrics Row ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Raw Accuracy", f"{acc * 100:.1f}%")
        col2.metric("Recall (Sensitivity)", f"{recall_score(y_test, y_test_pred) * 100:.1f}%")
        col3.metric("Precision", f"{precision_score(y_test, y_test_pred) * 100:.1f}%")
        col4.metric("F1 Score", f"{f1_score(y_test, y_test_pred) * 100:.1f}%")
        
        st.divider()
        
        # --- Visual Graphs Row ---
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            st.subheader("Confusion Matrix Heatmap")
            # Generate a beautiful Seaborn heatmap
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Healthy (0)', 'Cardiomegaly (1)'],
                        yticklabels=['Healthy (0)', 'Cardiomegaly (1)'])
            plt.ylabel('Actual Truth')
            plt.xlabel('Model Prediction')
            
            # Make the background transparent to match Streamlit's dark mode
            fig.patch.set_alpha(0.0) 
            ax.patch.set_alpha(0.0)
            
            st.pyplot(fig)
            
        with fig_col2:
            st.subheader("Why Accuracy is Misleading")
            st.write(f"**True Positives Detected:** {cm[1,1]} out of {np.sum(y_test == 1)}")
            st.write(f"**False Positives (False Alarms):** {cm[0,1]}")
            st.markdown("""
            By balancing the training data to a strict 50/50 split, we forced the K-Nearest Neighbor algorithm to stop blindly guessing 'Healthy'. 
            
            As a result, our **Recall** (ability to detect the disease) jumped to ~60%. However, because flattening a 2D X-ray into a 1D vector destroys the spatial structure of the heart, the model gets confused by background noise and triggers a massive amount of False Positives, dragging down the overall accuracy.
            """)
            
        st.info("Next Phase: Implementing Convolutional Neural Networks (CNNs) to preserve 2D spatial structures and eliminate False Positives.")

        
        # # 5. UI Display
        # st.success("Evaluation Complete!")
        
        # col1, col2 = st.columns(2)
        # col1.metric("Raw Accuracy", f"{acc * 100:.2f}%")
        # col2.metric("True Positives Detected", f"{cm[1,1]} / {np.sum(y_test == 1)}")
        
        # st.subheader("Confusion Matrix")
        # st.code(f"True Negatives:  {cm[0,0]} | False Positives: {cm[0,1]}\nFalse Negatives: {cm[1,0]}   | True Positives:  {cm[1,1]}")
        
        # st.info("Notice how classical ML heavily biases toward False Positives due to the loss of 2D spatial structures in the flattened array.")