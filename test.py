import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

from igsa_feature_selection.ImprovedGravitationalSearchAlgorithm import ImprovedGravitationalSearchAlgorithm

def evaluate_feature_selection(X, y, dataset_name):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    """
    Demonstrate feature selection process
    
    Parameters:
    - X: Input features
    - y: Target labels
    - dataset_name: Name of the dataset
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train without feature selection
    print(f"\n--- {dataset_name} Dataset ---")
    print("Original Feature Count:", X.shape[1])
    
    # Full feature classifier
    full_clf = KNeighborsClassifier(n_neighbors=1)
    full_clf.fit(X_train_scaled, y_train)
    full_pred = full_clf.predict(X_test_scaled)
    full_accuracy = accuracy_score(y_test, full_pred)
    print("Accuracy with All Features:", full_accuracy)
    
    # Initialize IGSA
    igsa = ImprovedGravitationalSearchAlgorithm(
        n_features=X.shape[1],  # Total number of features
        population_size=20,     # Number of candidate solutions
        max_iterations=100,     # Optimization iterations
        w=0.8                   # Weight for accuracy vs feature reduction
    )
    
    # Perform feature selection with timing
    start_time = time.time()
    selected_features = igsa.feature_selection(X_train_scaled, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print selected features and timing
    selected_feature_indices = np.where(selected_features)[0]
    print("Selected Feature Indices:", selected_feature_indices)
    print("Number of Selected Features:", np.sum(selected_features))
    print(f"Feature Selection Time: {elapsed_time:.2f} seconds")
    
    # Train with selected features
    X_train_selected = X_train_scaled[:, selected_features]
    X_test_selected = X_test_scaled[:, selected_features]
    
    selected_clf = KNeighborsClassifier(n_neighbors=1)
    selected_clf.fit(X_train_selected, y_train)
    selected_pred = selected_clf.predict(X_test_selected)
    selected_accuracy = accuracy_score(y_test, selected_pred)
    print("Accuracy with Selected Features:", selected_accuracy)
    
    # Compare feature reduction and accuracy
    print(f"Feature Reduction: {(1 - np.sum(selected_features)/X.shape[1])*100:.2f}%")
    print(f"Accuracy Change: {(selected_accuracy - full_accuracy)*100:.2f} percentage points")

def main():
    # Test on multiple datasets
    datasets = [
        load_breast_cancer(return_X_y=True),
        load_wine(return_X_y=True),
        load_iris(return_X_y=True)
    ]
    
    dataset_names = ['Breast Cancer', 'Wine', 'Iris']
    
    # Run feature selection on each dataset
    for (X, y), name in zip(datasets, dataset_names):
        evaluate_feature_selection(X, y, name)

if __name__ == "__main__":
    main()