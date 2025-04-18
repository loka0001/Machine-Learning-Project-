import random
import math
import matplotlib.pyplot as plt
import os

# Define features globally
NUMERIC_FEATURES = ['age', 'bmi', 'children']
CATEGORICAL_FEATURES = ['sex', 'smoker']

def read_csv(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        for line in f:
            row = []
            value = ''
            inside_quotes = False
            for c in line:
                if c == '"':
                    inside_quotes = not inside_quotes
                elif c == ',' and not inside_quotes:
                    row.append(value)
                    value = ''
                else:
                    value += c
            row.append(value.strip())
            data.append(row)
    return header, data

def normalize_features(X):
    """Normalize features using min-max scaling"""
    X = [[float(x) for x in row] for row in X]  # Convert to float
    n_features = len(X[0])
    mins = [min(col) for col in zip(*X)]
    maxs = [max(col) for col in zip(*X)]
    
    normalized = []
    for row in X:
        normalized_row = [(x - min_val) / (max_val - min_val + 1e-8) 
                         for x, min_val, max_val in zip(row, mins, maxs)]
        normalized.append(normalized_row)
    return normalized, mins, maxs

# Logistic Regression Classifier (no libraries)
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=32, 
                 lambda_reg=0.01, tolerance=1e-4):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg  # L2 regularization parameter
        self.tolerance = tolerance
        self.weights = None
        self.bias = 0
        self.costs = []  # Track cost history
        
    def sigmoid(self, z):
        """Numerically stable sigmoid function"""
        # Clip z to prevent overflow
        z = max(min(z, 500), -500)
        return 1.0 / (1.0 + pow(2.71828, -z))
    
    def compute_cost(self, X, y):
        """Compute binary cross-entropy loss with L2 regularization"""
        m = len(X)
        predictions = self.predict_proba(X)
        epsilon = 1e-15  # Prevent log(0)
        predictions = [max(min(p, 1 - epsilon), epsilon) for p in predictions]
        
        # Calculate log loss
        cost = (-1/m) * sum(
            y[i] * math.log(predictions[i]) + 
            (1 - y[i]) * math.log(1 - predictions[i]) 
            for i in range(m)
        )
        # Add L2 regularization term
        reg_term = (self.lambda_reg / (2*m)) * sum(w*w for w in self.weights)
        return cost + reg_term
    
    def train(self, X, y):
        X = normalize_features(X)[0]  # Normalize features
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        prev_cost = float('inf')
        
        for epoch in range(self.epochs):
            # Mini-batch gradient descent
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:min(i + self.batch_size, n_samples)]
                grad_w = [0.0] * n_features
                grad_b = 0.0
                
                for idx in batch_indices:
                    linear_model = sum(self.weights[j] * float(X[idx][j]) 
                                     for j in range(n_features)) + self.bias
                    y_pred = self.sigmoid(linear_model)
                    error = y_pred - float(y[idx])
                    
                    # Compute gradients
                    for j in range(n_features):
                        grad_w[j] += error * float(X[idx][j])
                    grad_b += error
                
                # Update weights with regularization
                batch_size = len(batch_indices)
                for j in range(n_features):
                    self.weights[j] -= (self.learning_rate * 
                                      (grad_w[j]/batch_size + 
                                       self.lambda_reg * self.weights[j]))
                self.bias -= self.learning_rate * (grad_b/batch_size)
            
            # Compute cost and check convergence
            try:
                current_cost = self.compute_cost(X, y)
                self.costs.append(current_cost)
                
                if abs(prev_cost - current_cost) < self.tolerance:
                    print(f"Converged at epoch {epoch}")
                    break
                    
                prev_cost = current_cost
            except (OverflowError, ValueError) as e:
                print(f"Numerical error at epoch {epoch}, continuing training")
                continue
    
    def predict_proba(self, X):
        """Predict probability of class 1"""
        return [self.sigmoid(sum(self.weights[j] * float(x[j]) 
                               for j in range(len(x))) + self.bias) for x in X]
    
    def predict(self, X):
        X = normalize_features(X)[0]  # Normalize using same scaling
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

# Linear Regression Class (no libraries)
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=32, 
                 lambda_reg=0.01, tolerance=1e-4):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.tolerance = tolerance
        self.weights = None
        self.bias = 0
        self.costs = []
        self.feature_means = None
        self.feature_stds = None
        
    def standardize_features(self, X):
        """Standardize features to have zero mean and unit variance"""
        X = [[float(x) for x in row] for row in X]
        if self.feature_means is None:  # Training phase
            # Calculate means
            self.feature_means = []
            for j in range(len(X[0])):
                col = [row[j] for row in X]
                self.feature_means.append(sum(col) / len(col))
            
            # Calculate standard deviations
            self.feature_stds = []
            for j in range(len(X[0])):
                col = [row[j] for row in X]
                mean = self.feature_means[j]
                variance = sum((x - mean) ** 2 for x in col) / len(col)
                self.feature_stds.append(math.sqrt(variance))
        
        # Apply standardization
        return [[(x - mean)/(std + 1e-8) 
                 for x, mean, std in zip(row, self.feature_means, self.feature_stds)]
                for row in X]
    
    def compute_cost(self, X, y):
        """Compute MSE loss with L2 regularization"""
        m = len(X)
        predictions = self.predict(X)
        mse = sum((pred - float(actual))**2 
                 for pred, actual in zip(predictions, y)) / (2*m)
        reg_term = (self.lambda_reg/(2*m)) * sum(w*w for w in self.weights)
        return mse + reg_term
    
    def train(self, X, y):
        X = self.standardize_features(X)
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        prev_cost = float('inf')
        y = [float(val) for val in y]  # Convert y to float
        
        for epoch in range(self.epochs):
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:min(i + self.batch_size, n_samples)]
                grad_w = [0.0] * n_features
                grad_b = 0.0
                
                for idx in batch_indices:
                    prediction = sum(self.weights[j] * X[idx][j] 
                                   for j in range(n_features)) + self.bias
                    error = prediction - y[idx]
                    
                    for j in range(n_features):
                        grad_w[j] += error * X[idx][j]
                    grad_b += error
                
                batch_size = len(batch_indices)
                for j in range(n_features):
                    self.weights[j] -= (self.learning_rate * 
                                      (grad_w[j]/batch_size + 
                                       self.lambda_reg * self.weights[j]))
                self.bias -= self.learning_rate * (grad_b/batch_size)
            
            current_cost = self.compute_cost(X, y)
            self.costs.append(current_cost)
            
            if abs(prev_cost - current_cost) < self.tolerance:
                print(f"Converged at epoch {epoch}")
                break
                
            prev_cost = current_cost
    
    def predict(self, X):
        X = self.standardize_features(X)
        return [sum(self.weights[j] * x[j] for j in range(len(x))) + self.bias 
                for x in X]

# Accuracy and Confusion Matrix
def evaluate(predictions, actual):
    correct = sum([1 for p, a in zip(predictions, actual) if p == a])
    accuracy = correct / len(actual)

    TP = sum([1 for p, a in zip(predictions, actual) if p == 1 and a == 1])
    TN = sum([1 for p, a in zip(predictions, actual) if p == 0 and a == 0])
    FP = sum([1 for p, a in zip(predictions, actual) if p == 1 and a == 0])
    FN = sum([1 for p, a in zip(predictions, actual) if p == 0 and a == 1])

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(f"TP: {TP} | FP: {FP}")
    print(f"FN: {FN} | TN: {TN}")

# Mean Absolute Error for regression

def mean_absolute_error(predictions, actual):
    total_error = sum([abs(p - float(a)) for p, a in zip(predictions, actual)])
    return total_error / len(actual)

def print_section(title):
    """Print a section title in a formatted way"""
    print("\n" + "="*50)
    print(f"{title:^50}")
    print("="*50)

def print_step(step):
    """Print a step in a formatted way"""
    print(f"\n➤ {step}")

def print_metric(name, value, is_currency=False):
    """Print a metric in a formatted way"""
    if is_currency:
        print(f"  • {name:<25} ${value:,.2f}")
    else:
        print(f"  • {name:<25} {value:.4f}")

def read_diabetes_data(path):
    """Read and preprocess diabetes dataset"""
    header, data = read_csv(path)
    X, y = [], []
    for row in data[1:]:  # Skip header
        try:
            features = [float(val) if val != '' else 0 for val in row[:-1]]  # Convert all features to float
            X.append(features)
            y.append(int(row[-1]))  # Convert outcome to int
        except Exception as e:
            print(f"  • Error processing row: {e}")
            continue
    return X, y, header[:-1]  # Return features without 'Outcome'

def visualize_regression(X, y, model, feature_idx=2, feature_name="BMI", save_path=None):
    """
    Visualize the regression line for a single feature and save it as an image.
    Args:
        X: List of feature vectors
        y: List of target values
        model: Trained model with predict method
        feature_idx: Index of the feature to plot
        feature_name: Name of the feature being plotted
        save_path: Path to save the image
    """
    # Extract the chosen feature
    x_plot = [x[feature_idx] for x in X]
    
    # Create evenly spaced points for the line
    x_min, x_max = min(x_plot), max(x_plot)
    x_range = []
    step = (x_max - x_min) / 100
    current = x_min
    while current <= x_max:
        x_range.append(current)
        current += step
    
    # Create feature vectors for prediction
    X_plot = []
    for x in x_range:
        features = [0] * len(X[0])  # Initialize with median values
        for j in range(len(X[0])):
            if j == feature_idx:
                features[j] = x
            else:
                # Use median value for other features
                values = [X[i][j] for i in range(len(X))]
                features[j] = sorted(values)[len(values)//2]
        X_plot.append(features)
    
    # Get predictions
    y_pred = model.predict(X_plot)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    plt.scatter(x_plot, y, color='#2E86C1', alpha=0.5, s=50, label='Data Points')
    
    # Plot regression line
    plt.plot(x_range, y_pred, color='#E74C3C', linewidth=3, label='Linear Regression')
    
    plt.xlabel(feature_name.upper(), fontsize=12, fontweight='bold')
    plt.ylabel('Insurance Cost ($)', fontsize=12, fontweight='bold')
    plt.title(f'Linear Regression: {feature_name.upper()} and Insurance Cost', 
             fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_mlpy{ext}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  • Saved plot to {save_path}")

def visualize_classification(X, y, model, feature1_idx=1, feature2_idx=5,
                           feature1_name="Glucose", feature2_name="BMI", save_path=None):
    """
    Visualize the decision boundary for logistic regression using two features and save it as an image.
    Args:
        X: List of feature vectors
        y: List of target values
        model: Trained model with predict method
        feature1_idx, feature2_idx: Indices of features to plot
        feature1_name, feature2_name: Names of features being plotted
        save_path: Path to save the image
    """
    # Extract the two chosen features
    x1 = [x[feature1_idx] for x in X]
    x2 = [x[feature2_idx] for x in X]
    
    # Create mesh grid
    x1_min, x1_max = min(x1) - 1, max(x1) + 1
    x2_min, x2_max = min(x2) - 1, max(x2) + 1
    step = 0.5  # Increased step size for better performance
    
    # Create grid points
    x1_grid = []
    current = x1_min
    while current <= x1_max:
        x1_grid.append(current)
        current += step
        
    x2_grid = []
    current = x2_min
    while current <= x2_max:
        x2_grid.append(current)
        current += step
    
    # Create predictions for each grid point
    Z = []
    for x2_val in x2_grid:
        row = []
        for x1_val in x1_grid:
            # Create feature vector with median values
            features = []
            for j in range(len(X[0])):
                if j == feature1_idx:
                    features.append(x1_val)
                elif j == feature2_idx:
                    features.append(x2_val)
                else:
                    # Use median value for other features
                    values = [X[i][j] for i in range(len(X))]
                    features.append(sorted(values)[len(values)//2])
            
            # Get prediction
            pred = model.predict([features])[0]
            row.append(pred)
        Z.append(row)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Create meshgrid for plotting
    XX1 = [[x1 for x1 in x1_grid] for _ in x2_grid]
    XX2 = [[x2 for _ in x1_grid] for x2 in x2_grid]
    
    # Plot decision boundary
    plt.contourf(XX1, XX2, Z, alpha=0.4, cmap='RdYlBu', levels=[-0.5, 0.5, 1.5])
    
    # Plot the actual data points
    for label in [0, 1]:
        mask = [y_i == label for y_i in y]
        plt.scatter([x1[i] for i in range(len(x1)) if mask[i]],
                   [x2[i] for i in range(len(x2)) if mask[i]],
                   label=f'Class {label}',
                   alpha=0.6)
    
    plt.xlabel(feature1_name, fontsize=12, fontweight='bold')
    plt.ylabel(feature2_name, fontsize=12, fontweight='bold')
    plt.title(f'Diabetes Classification\n{feature1_name} vs {feature2_name}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_mlpy{ext}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  • Saved plot to {save_path}")

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
        print("Created 'plots' directory")

    # Insurance Data (Linear Regression)
    print_section("Insurance Cost Prediction Model")
    print_step("Loading insurance data...")
    header, data = read_csv("insurance.csv")
    print(f"  • Loaded {len(data)} records from insurance.csv")

    def preprocess_insurance_data(data):
        X, y = [], []
        for row in data:
            try:
                # Convert features
                age = float(row[0])
                sex = 1 if row[1].strip().lower() == 'male' else 0
                bmi = float(row[2])
                children = float(row[3])
                smoker = 1 if row[4].strip().lower() == 'yes' else 0
                
                # Combine features (excluding region)
                features = [age, sex, bmi, children, smoker]
                X.append(features)
                
                # Target variable (charges)
                y.append(float(row[6]))
            except Exception as e:
                print(f"  • Error processing row: {e}")
                continue
        return X, y

    # Process insurance data
    print_step("Preprocessing insurance data...")
    data = data[1:]  # Remove header row
    X_insurance, y_insurance = preprocess_insurance_data(data)
    print(f"  • Successfully processed {len(X_insurance)} insurance records")
    
    # Split insurance data
    print_step("Splitting insurance data into train/test sets...")
    train_size = int(0.8 * len(X_insurance))
    indices = list(range(len(X_insurance)))
    random.shuffle(indices)
    
    X_train_insurance = [X_insurance[i] for i in indices[:train_size]]
    y_train_insurance = [y_insurance[i] for i in indices[:train_size]]
    X_test_insurance = [X_insurance[i] for i in indices[train_size:]]
    y_test_insurance = [y_insurance[i] for i in indices[train_size:]]
    
    # Train and evaluate linear regression
    print_section("Training Linear Regression Model")
    print_step("Initializing model...")
    reg = LinearRegression(learning_rate=0.001, epochs=1000, batch_size=32)
    print("  • Learning rate: 0.001")
    print("  • Epochs: 1000")
    print("  • Batch size: 32")
    
    print_step("Training model...")
    reg.train(X_train_insurance, y_train_insurance)
    
    # Make insurance predictions
    print_step("Making insurance predictions...")
    train_predictions = reg.predict(X_train_insurance)
    test_predictions = reg.predict(X_test_insurance)
    
    # Print insurance results
    print_step("Insurance Training Results:")
    print_metric("Mean Absolute Error", mean_absolute_error(train_predictions, y_train_insurance), True)
    
    print_step("Insurance Test Results:")
    print_metric("Mean Absolute Error", mean_absolute_error(test_predictions, y_test_insurance), True)
    
    # Visualize insurance regression results
    print_section("Visualization - Insurance Model")
    print_step("Saving regression plots...")
    visualize_regression(X_train_insurance, y_train_insurance, reg,
                        feature_idx=2, feature_name="BMI",
                        save_path='plots/insurance_bmi.png')
    visualize_regression(X_train_insurance, y_train_insurance, reg,
                        feature_idx=0, feature_name="Age",
                        save_path='plots/insurance_age.png')

    # Diabetes Data (Logistic Regression)
    print_section("Diabetes Prediction Model")
    print_step("Loading diabetes data...")
    X_diabetes, y_diabetes, feature_names = read_diabetes_data("diabetes-dataset.csv")
    print(f"  • Loaded {len(X_diabetes)} diabetes records")
    
    # Split diabetes data
    print_step("Splitting diabetes data into train/test sets...")
    train_size = int(0.8 * len(X_diabetes))
    indices = list(range(len(X_diabetes)))
    random.shuffle(indices)
    
    X_train_diabetes = [X_diabetes[i] for i in indices[:train_size]]
    y_train_diabetes = [y_diabetes[i] for i in indices[:train_size]]
    X_test_diabetes = [X_diabetes[i] for i in indices[train_size:]]
    y_test_diabetes = [y_diabetes[i] for i in indices[train_size:]]
    
    print(f"  • Training set size: {len(X_train_diabetes)}")
    print(f"  • Test set size: {len(X_test_diabetes)}")
    
    # Train and evaluate logistic regression
    print_section("Training Logistic Regression Model")
    print_step("Initializing model...")
    clf = LogisticRegression(learning_rate=0.01, epochs=1000, batch_size=32)
    print("  • Learning rate: 0.01")
    print("  • Epochs: 1000")
    print("  • Batch size: 32")
    
    print_step("Training model...")
    clf.train(X_train_diabetes, y_train_diabetes)
    
    # Make diabetes predictions
    print_step("Making diabetes predictions...")
    train_pred_diabetes = clf.predict(X_train_diabetes)
    test_pred_diabetes = clf.predict(X_test_diabetes)
    
    # Print diabetes results
    print_step("Diabetes Training Results:")
    evaluate(train_pred_diabetes, y_train_diabetes)
    
    print_step("Diabetes Test Results:")
    evaluate(test_pred_diabetes, y_test_diabetes)
    
    # Visualize diabetes classification results
    print_section("Visualization - Diabetes Model")
    print_step("Saving classification plots...")
    visualize_classification(X_train_diabetes, y_train_diabetes, clf,
                           feature1_idx=1, feature2_idx=5,
                           feature1_name="Glucose", feature2_name="BMI",
                           save_path='plots/diabetes_glucose_bmi.png')
    visualize_classification(X_train_diabetes, y_train_diabetes, clf,
                           feature1_idx=7, feature2_idx=1,
                           feature1_name="Age", feature2_name="Glucose",
                           save_path='plots/diabetes_age_glucose.png')
    
    # Print feature importance for both models
    print_section("Feature Importance Analysis")
    
    # Linear Regression (Insurance)
    