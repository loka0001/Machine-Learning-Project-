import csv
import random
import math
from collections import defaultdict

def load_data(filename):
    """Load data from CSV file."""
    try:
        data = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                if len(row) < 2:  # Check if row has enough elements
                    continue
                # Convert outcome to class 1 or 2
                row[-1] = '1' if row[-1] == '0' else '2'
                data.append([float(x) for x in row])
        return data, headers
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def preprocess_data(data):
    """Preprocess data by handling missing values."""
    try:
        if not data or len(data) == 0:
            return None
            
        # Convert zeros to None for specific columns
        zero_cols = [1, 2, 3, 4, 5]  # Glucose, BloodPressure, SkinThickness, Insulin, BMI
        data = [row[:] for row in data]  # Create a deep copy
        
        # Calculate medians for each column
        medians = []
        for col in range(len(data[0])):
            values = [row[col] for row in data if row[col] is not None and row[col] != 0]
            if values:
                values.sort()
                medians.append(values[len(values)//2])
            else:
                medians.append(0)
        
        # Replace zeros and None with medians
        for row in data:
            for col in range(len(row)):
                if col in zero_cols and (row[col] == 0 or row[col] is None):
                    row[col] = medians[col]
        
        return data
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        return None

def standardize_data(data):
    """Standardize the data (z-score normalization)."""
    try:
        if not data or len(data) == 0:
            return None
            
        # Calculate mean and standard deviation for each feature
        means = []
        stds = []
        for col in range(len(data[0]) - 1):  # Exclude the outcome column
            values = [row[col] for row in data if row[col] is not None]
            if not values:
                return None
            mean = sum(values) / len(values)
            std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
            means.append(mean)
            stds.append(std)
        
        # Standardize the data
        standardized = []
        for row in data:
            if len(row) < len(means) + 1:  # Check if row has enough elements
                continue
            new_row = [(row[i] - means[i]) / stds[i] for i in range(len(row) - 1)]
            new_row.append(row[-1])  # Keep the outcome unchanged
            standardized.append(new_row)
        
        return standardized
    except Exception as e:
        print(f"Error standardizing data: {str(e)}")
        return None

def train_test_split(data, test_size=0.2):
    """Split data into training and testing sets."""
    try:
        if not data or len(data) == 0:
            return None, None
            
        random.shuffle(data)
        split_idx = int(len(data) * (1 - test_size))
        return data[:split_idx], data[split_idx:]
    except Exception as e:
        print(f"Error splitting data: {str(e)}")
        return None, None

def rbf_kernel(x1, x2, gamma=1.0):
    """Calculate RBF kernel between two points."""
    try:
        if len(x1) != len(x2):
            return 0.0
        return math.exp(-gamma * sum((a - b) ** 2 for a, b in zip(x1, x2)))
    except Exception:
        return 0.0

def compute_gradient(X, y, alpha, b, i, kernel_matrix, C=1.0):
    """Compute gradient for a single sample."""
    try:
        if i >= len(X) or i >= len(y) or i >= len(alpha):
            return 0.0
        pred = sum(alpha[j] * y[j] * kernel_matrix[i][j] for j in range(len(X))) + b
        error = pred - y[i]
        return error
    except Exception:
        return 0.0

def update_parameters(alpha, b, i, j, error_i, error_j, kernel_matrix, y, C=1.0):
    """Update SVM parameters using gradient information."""
    try:
        if i >= len(alpha) or j >= len(alpha) or i >= len(y) or j >= len(y):
            return alpha, b, False
            
        old_alpha_i = alpha[i]
        old_alpha_j = alpha[j]
        
        if y[i] != y[j]:
            L = max(0, alpha[j] - alpha[i])
            H = min(C, C + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[i] + alpha[j] - C)
            H = min(C, alpha[i] + alpha[j])
        
        if L == H:
            return alpha, b, False
        
        eta = 2 * kernel_matrix[i][j] - kernel_matrix[i][i] - kernel_matrix[j][j]
        if eta >= 0:
            return alpha, b, False
        
        alpha[j] -= y[j] * (error_i - error_j) / eta
        alpha[j] = max(L, min(H, alpha[j]))
        
        if abs(alpha[j] - old_alpha_j) < 1e-5:
            return alpha, b, False
        
        alpha[i] += y[i] * y[j] * (old_alpha_j - alpha[j])
        
        b1 = b - error_i - y[i] * (alpha[i] - old_alpha_i) * kernel_matrix[i][i] - y[j] * (alpha[j] - old_alpha_j) * kernel_matrix[i][j]
        b2 = b - error_j - y[i] * (alpha[i] - old_alpha_i) * kernel_matrix[i][j] - y[j] * (alpha[j] - old_alpha_j) * kernel_matrix[j][j]
        
        if 0 < alpha[i] < C:
            b = b1
        elif 0 < alpha[j] < C:
            b = b2
        else:
            b = (b1 + b2) / 2
        
        return alpha, b, True
    except Exception:
        return alpha, b, False

def train_svm_batch(X, y, C=1.0, gamma=1.0, max_iter=1000, tol=0.001):
    """Train SVM using batch gradient descent with improved accuracy."""
    try:
        if not X or not y or len(X) != len(y):
            return None, None
            
        n_samples = len(X)
        alpha = [0.0] * n_samples
        b = 0.0
        
        # Initialize momentum
        momentum_alpha = [0.0] * n_samples
        momentum_b = 0.0
        momentum_factor = 0.9
        
        # Initialize learning rate
        learning_rate = 0.01
        min_learning_rate = 0.0001
        learning_rate_decay = 0.95
        
        # Precompute kernel matrix
        kernel_matrix = [[0.0] * n_samples for _ in range(n_samples)]
        for i in range(n_samples):
            for j in range(i, n_samples):
                kernel_matrix[i][j] = kernel_matrix[j][i] = rbf_kernel(X[i], X[j], gamma)
        
        # Initialize best parameters
        best_alpha = alpha[:]
        best_b = b
        best_accuracy = 0.0
        no_improvement_count = 0
        max_no_improvement = 10
        
        for iteration in range(max_iter):
            alpha_prev = alpha[:]
            b_prev = b
            num_changed = 0
            
            # Compute gradients for all samples
            gradients = {}
            for i in range(n_samples):
                gradients[i] = compute_gradient(X, y, alpha, b, i, kernel_matrix, C)
            
            # Calculate total gradient magnitude
            total_gradient = sum(abs(g) for g in gradients.values())
            if total_gradient < tol:
                break
            
            # Update parameters with momentum
            for i in range(n_samples):
                if (y[i] * gradients[i] < -tol and alpha[i] < C) or (y[i] * gradients[i] > tol and alpha[i] > 0):
                    # Apply momentum
                    momentum_alpha[i] = momentum_factor * momentum_alpha[i] + learning_rate * gradients[i]
                    alpha[i] -= momentum_alpha[i]
                    
                    # Project alpha back to feasible region
                    alpha[i] = max(0, min(C, alpha[i]))
                    num_changed += 1
            
            # Update bias with momentum
            bias_gradient = sum(gradients.values()) / n_samples
            momentum_b = momentum_factor * momentum_b + learning_rate * bias_gradient
            b -= momentum_b
            
            # Adaptive learning rate
            if num_changed == 0:
                learning_rate *= learning_rate_decay
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate
            
            # Early stopping with best model preservation
            if iteration % 10 == 0:  # Check accuracy every 10 iterations
                current_accuracy = evaluate_accuracy(X, y, alpha, b, kernel_matrix)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_alpha = alpha[:]
                    best_b = b
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= max_no_improvement:
                        break
            
            # Check convergence
            alpha_diff = sum(abs(a1 - a2) for a1, a2 in zip(alpha, alpha_prev))
            b_diff = abs(b - b_prev)
            if alpha_diff + b_diff < tol:
                break
        
        # Return best model found
        return best_alpha, best_b
    except Exception as e:
        print(f"Error in batch training: {str(e)}")
        return None, None

def evaluate_accuracy(X, y, alpha, b, kernel_matrix):
    """Evaluate accuracy of current model."""
    try:
        correct = 0
        for i in range(len(X)):
            pred = sum(alpha[j] * y[j] * kernel_matrix[i][j] for j in range(len(X))) + b
            if (pred > 0 and y[i] > 0) or (pred <= 0 and y[i] <= 0):
                correct += 1
        return correct / len(X)
    except Exception:
        return 0.0

def train_svm_minibatch(X, y, C=1.0, gamma=1.0, max_iter=1000, tol=0.001, batch_size=32):
    """Train SVM using mini-batch gradient descent."""
    try:
        if not X or not y or len(X) != len(y):
            return None, None
            
        n_samples = len(X)
        alpha = [0.0] * n_samples
        b = 0.0
        
        # Precompute kernel matrix
        kernel_matrix = [[0.0] * n_samples for _ in range(n_samples)]
        for i in range(n_samples):
            for j in range(i, n_samples):
                kernel_matrix[i][j] = kernel_matrix[j][i] = rbf_kernel(X[i], X[j], gamma)
        
        for _ in range(max_iter):
            alpha_prev = alpha[:]
            num_changed = 0
            
            # Create mini-batches
            indices = list(range(n_samples))
            random.shuffle(indices)
            batches = [indices[i:i + batch_size] for i in range(0, n_samples, batch_size)]
            
            for batch in batches:
                if not batch:  # Skip empty batches
                    continue
                    
                # Compute gradients for batch
                gradients = {}
                for i in batch:
                    gradients[i] = compute_gradient(X, y, alpha, b, i, kernel_matrix, C)
                
                # Update parameters for batch
                for i in batch:
                    if i not in gradients:
                        continue
                        
                    if (y[i] * gradients[i] < -tol and alpha[i] < C) or (y[i] * gradients[i] > tol and alpha[i] > 0):
                        # Choose j from the same batch
                        j = random.choice(batch)
                        while j == i:
                            j = random.choice(batch)
                            
                        if j not in gradients:
                            gradients[j] = compute_gradient(X, y, alpha, b, j, kernel_matrix, C)
                            
                        alpha, b, changed = update_parameters(alpha, b, i, j, gradients[i], gradients[j], kernel_matrix, y, C)
                        if changed:
                            num_changed += 1
            
            if num_changed == 0:
                break
        
        return alpha, b
    except Exception as e:
        print(f"Error in mini-batch training: {str(e)}")
        return None, None

def train_svm_stochastic(X, y, C=1.0, gamma=1.0, max_iter=1000, tol=0.001):
    """Train SVM using stochastic gradient descent."""
    try:
        if not X or not y or len(X) != len(y):
            return None, None
            
        n_samples = len(X)
        alpha = [0.0] * n_samples
        b = 0.0
        
        # Precompute kernel matrix
        kernel_matrix = [[0.0] * n_samples for _ in range(n_samples)]
        for i in range(n_samples):
            for j in range(i, n_samples):
                kernel_matrix[i][j] = kernel_matrix[j][i] = rbf_kernel(X[i], X[j], gamma)
        
        for _ in range(max_iter):
            alpha_prev = alpha[:]
            num_changed = 0
            
            # Randomly select samples
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            for i in indices:
                gradient = compute_gradient(X, y, alpha, b, i, kernel_matrix, C)
                
                if (y[i] * gradient < -tol and alpha[i] < C) or (y[i] * gradient > tol and alpha[i] > 0):
                    j = random.randint(0, n_samples - 1)
                    while j == i:
                        j = random.randint(0, n_samples - 1)
                    
                    gradient_j = compute_gradient(X, y, alpha, b, j, kernel_matrix, C)
                    alpha, b, changed = update_parameters(alpha, b, i, j, gradient, gradient_j, kernel_matrix, y, C)
                    if changed:
                        num_changed += 1
            
            if num_changed == 0:
                break
        
        return alpha, b
    except Exception as e:
        print(f"Error in stochastic training: {str(e)}")
        return None, None

def predict_svm(X_train, y_train, X_test, alpha, b, gamma=1.0):
    """Make predictions using trained SVM."""
    try:
        if not X_train or not y_train or not X_test or not alpha:
            return []
            
        predictions = []
        for x in X_test:
            pred = sum(alpha[i] * y_train[i] * rbf_kernel(x, X_train[i], gamma) for i in range(len(X_train))) + b
            predictions.append(1 if pred > 0 else 2)
        return predictions
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return []

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy of predictions."""
    try:
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return 0.0
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true)
    except Exception:
        return 0.0

def main():
    try:
        print("Loading diabetes dataset...")
        data, headers = load_data('diabetes-dataset.csv')
        if data is None or not data:
            print("Error: No data loaded")
            return
        
        print("\nPreprocessing data...")
        data = preprocess_data(data)
        if data is None or not data:
            print("Error: Data preprocessing failed")
            return
        
        print("\nStandardizing data...")
        data = standardize_data(data)
        if data is None or not data:
            print("Error: Data standardization failed")
            return
        
        print("\nSplitting data into training and testing sets...")
        train_data, test_data = train_test_split(data)
        if train_data is None or test_data is None or not train_data or not test_data:
            print("Error: Data splitting failed")
            return
        
        # Prepare features and labels
        X_train = [point[:-1] for point in train_data]
        y_train = [point[-1] for point in train_data]
        X_test = [point[:-1] for point in test_data]
        y_test = [point[-1] for point in test_data]
        
        # Convert labels to -1 and 1 for SVM
        y_train_svm = [1 if y == 1 else -1 for y in y_train]
        
        # Train with different methods
        methods = {
            "Batch": train_svm_batch,
            "Mini-batch": train_svm_minibatch,
            "Stochastic": train_svm_stochastic
        }
        
        for method_name, train_func in methods.items():
            print(f"\nTraining SVM using {method_name} gradient descent...")
            alpha, b = train_func(X_train, y_train_svm)
            
            if alpha is None or b is None:
                print(f"Error: {method_name} training failed")
                continue
            
            print(f"\nMaking predictions with {method_name}...")
            y_pred = predict_svm(X_train, y_train_svm, X_test, alpha, b)
            
            if not y_pred:
                print(f"Error: {method_name} prediction failed")
                continue
            
            # Calculate class distribution
            total_points = len(y_pred)
            class_1_count = sum(1 for pred in y_pred if pred == 1)
            class_2_count = sum(1 for pred in y_pred if pred == 2)
            class_1_percentage = (class_1_count / total_points) * 100
            class_2_percentage = (class_2_count / total_points) * 100
            
            print(f"\nPrediction Distribution ({method_name}):")
            print(f"Total points: {total_points}")
            print(f"Class 1: {class_1_count} points ({class_1_percentage:.2f}%)")
            print(f"Class 2: {class_2_count} points ({class_2_percentage:.2f}%)")
            
            # Evaluate model
            accuracy = calculate_accuracy(y_test, y_pred)
            print(f"\nModel Accuracy ({method_name}): {accuracy:.2f}")
            
            # Print confusion matrix
            print(f"\nConfusion Matrix ({method_name}):")
            confusion = defaultdict(int)
            for true, pred in zip(y_test, y_pred):
                confusion[(true, pred)] += 1
            print("True\\Pred | Class 1 | Class 2")
            print("----------------------------")
            print(f"Class 1   | {confusion[(1,1)]} | {confusion[(1,2)]}")
            print(f"Class 2   | {confusion[(2,1)]} | {confusion[(2,2)]}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 