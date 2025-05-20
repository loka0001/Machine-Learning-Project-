import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Load the diabetes dataset
data = pd.read_csv('diabetes.csv')

# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create and train SVM with different kernels
kernels = {
    'Quadratic': 'poly',  # polynomial with degree=2 (quadratic)
    'Polynomial (degree=2)': 'poly',  # Same as quadratic but explicitly named
    'RBF': 'rbf'  # Radial Basis Function
}

# Parameters for kernels
params = {
    'Quadratic': {'degree': 2, 'coef0': 1},
    'Polynomial (degree=2)': {'degree': 2, 'coef0': 1},
    'RBF': {}
}

# Dictionary to store models and their accuracy
models = {}
accuracies = {}

# Train models with different kernels
for name, kernel in kernels.items():
    print(f"Training SVM with {name} kernel...")
    model = SVC(kernel=kernel, **params[name])
    model.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store model and accuracy
    models[name] = model
    accuracies[name] = accuracy
    
    print(f"{name} kernel accuracy: {accuracy:.4f}")

# Visualize the results (accuracies)
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'])
plt.xlabel('Kernel Type')
plt.ylabel('Accuracy')
plt.title('SVM Accuracy with Different Kernels')
plt.ylim(0, 1)  # Accuracy is between 0 and 1
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.savefig('kernel_accuracies.png')
plt.show()

# Visualize decision boundaries (using PCA for dimensionality reduction)
print("Visualizing decision boundaries...")

# Apply PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split the reduced data
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Create a mesh grid for visualization
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plot the decision boundaries for each kernel
plt.figure(figsize=(15, 10))
i = 1
for name, kernel in kernels.items():
    # Train model on PCA-reduced data
    model_pca = SVC(kernel=kernel, **params[name])
    model_pca.fit(X_pca_train, y_pca_train)
    
    # Predict using the mesh grid
    Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.subplot(1, 3, i)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    # Plot the training points
    plt.scatter(X_pca_train[:, 0], X_pca_train[:, 1], c=y_pca_train, 
                cmap=plt.cm.coolwarm, edgecolors='k')
    
    # Calculate and display accuracy on PCA-transformed data
    y_pca_pred = model_pca.predict(X_pca_test)
    pca_accuracy = accuracy_score(y_pca_test, y_pca_pred)
    
    plt.title(f"{name} Kernel\nAccuracy: {pca_accuracy:.4f}")
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    i += 1

plt.tight_layout()
plt.savefig('kernel_decision_boundaries.png')
plt.show()

# Print final comparison
print("\nKernel Comparison:")
for name, accuracy in accuracies.items():
    print(f"{name} Kernel: {accuracy:.4f}")
    
print("\nAnalysis completed. Check the generated visualization files.") 