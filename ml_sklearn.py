import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Define features globally
INSURANCE_NUMERIC = ['age', 'bmi', 'children']
INSURANCE_CATEGORICAL = ['sex', 'smoker']

DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

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

def load_insurance_data(path):
    """Load and preprocess the insurance data"""
    print_step("Loading insurance data...")
    data = pd.read_csv(path)
    print(f"  • Loaded {len(data)} records from {path}")
    
    # Drop region column
    data = data.drop('region', axis=1)
    print(f"  • Dropped 'region' column")
    
    # Separate features and target
    X = data.drop('charges', axis=1)
    y = data['charges']
    print(f"  • Features shape: {X.shape}")
    print(f"  • Target shape: {y.shape}")
    
    return X, y

def load_diabetes_data(path):
    """Load and preprocess the diabetes data"""
    print_step("Loading diabetes data...")
    data = pd.read_csv(path)
    print(f"  • Loaded {len(data)} records from {path}")
    
    # Separate features and target
    X = data[DIABETES_FEATURES]
    y = data['Outcome']
    
    # Replace zeros with NaN and then fill with median for specific columns
    columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in columns_to_fix:
        X[column] = X[column].replace(0, np.nan)
        X[column] = X[column].fillna(X[column].median())
    
    print(f"  • Features shape: {X.shape}")
    print(f"  • Target shape: {y.shape}")
    
    return X, y

def create_insurance_pipeline():
    """Create a preprocessing pipeline for insurance data"""
    print_step("Creating insurance preprocessing pipeline...")
    print(f"  • Numeric features: {', '.join(INSURANCE_NUMERIC)}")
    print(f"  • Categorical features: {', '.join(INSURANCE_CATEGORICAL)}")
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, INSURANCE_NUMERIC),
            ('cat', categorical_transformer, INSURANCE_CATEGORICAL)
        ])
    
    return preprocessor

def create_diabetes_pipeline():
    """Create a preprocessing pipeline for diabetes data"""
    print_step("Creating diabetes preprocessing pipeline...")
    return StandardScaler()

def train_and_evaluate_regression(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a regression model"""
    print_section(f"Training {name}")
    
    pipeline = Pipeline([
        ('preprocessor', create_insurance_pipeline()),
        ('regressor', model)
    ])
    
    print_step("Fitting model...")
    pipeline.fit(X_train, y_train)
    
    print_step("Making predictions...")
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    print_step("Training Results:")
    print_metric("Mean Absolute Error", mean_absolute_error(y_train, y_pred_train), True)
    print_metric("R² Score", r2_score(y_train, y_pred_train))
    
    print_step("Test Results:")
    print_metric("Mean Absolute Error", mean_absolute_error(y_test, y_pred_test), True)
    print_metric("R² Score", r2_score(y_test, y_pred_test))
    
    return pipeline

def train_and_evaluate_classification(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a classification model"""
    print_section(f"Training {name}")
    
    pipeline = Pipeline([
        ('preprocessor', create_diabetes_pipeline()),
        ('classifier', model)
    ])
    
    print_step("Fitting model...")
    pipeline.fit(X_train, y_train)
    
    print_step("Making predictions...")
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    def print_classification_metrics(y_true, y_pred, dataset=""):
        print(f"\n{dataset} Results:")
        print_metric("Accuracy", accuracy_score(y_true, y_pred))
        print_metric("Precision", precision_score(y_true, y_pred))
        print_metric("Recall", recall_score(y_true, y_pred))
        print_metric("F1 Score", f1_score(y_true, y_pred))
        
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(f"TP: {cm[1,1]} | FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]} | TN: {cm[0,0]}")
    
    print_classification_metrics(y_train, y_pred_train, "Training")
    print_classification_metrics(y_test, y_pred_test, "Test")
    
    return pipeline

def visualize_regression_scatter(pipeline, X, y, feature_name="bmi", save_path=None):
    """
    Create a scatter plot with fitted line for regression analysis.
    Args:
        pipeline: Trained sklearn pipeline
        X: DataFrame of features
        y: Series of target values
        feature_name: Name of the feature to plot
        save_path: Path to save the image
    """
    plt.figure(figsize=(12, 8))
    
    # Get feature data after preprocessing
    X_transformed = pipeline.named_steps['preprocessor'].transform(X)
    feature_idx = X.columns.get_loc(feature_name)
    
    # Get predictions for the entire range
    X_plot = X.copy()
    x_range = np.linspace(X[feature_name].min(), X[feature_name].max(), 100)
    
    # Create scatter plot of actual data
    plt.scatter(X[feature_name], y, color='#2E86C1', alpha=0.5, s=50, label='Data Points')
    
    # Generate predictions for the line
    predictions = []
    for x_val in x_range:
        X_temp = X_plot.copy()
        X_temp[feature_name] = x_val
        pred = pipeline.predict(X_temp)
        predictions.append(np.mean(pred))
    
    # Plot fitted line
    plt.plot(x_range, predictions, color='#E74C3C', linewidth=3, label='Linear Regression')
    
    plt.xlabel(feature_name.upper(), fontsize=12, fontweight='bold')
    plt.ylabel('Insurance Cost ($)', fontsize=12, fontweight='bold')
    plt.title(f'Linear Regression: {feature_name.upper()} and Insurance Cost', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add grid with lower opacity
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend with better placement
    plt.legend(fontsize=10, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        # Add source file indication to filename
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_sklearn{ext}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  • Saved plot to {save_path}")

def visualize_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Create a heatmap visualization of the confusion matrix.
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the image
    """
    plt.figure(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        # Add source file indication to filename
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_sklearn{ext}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  • Saved plot to {save_path}")

def visualize_all_regression_features(pipeline, X, y, save_dir='plots'):
    """
    Create scatter plots with fitted lines for all numeric features in the regression analysis.
    Args:
        pipeline: Trained sklearn pipeline
        X: DataFrame of features
        y: Series of target values
        save_dir: Directory to save the images
    """
    print_step("Creating regression plots for all numeric features...")
    
    # Create subplots for all numeric features
    numeric_features = INSURANCE_NUMERIC
    n_features = len(numeric_features)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    plt.figure(figsize=(15, 6 * n_rows))
    
    for idx, feature in enumerate(numeric_features, 1):
        plt.subplot(n_rows, n_cols, idx)
        
        # Get predictions for the entire range
        X_plot = X.copy()
        x_range = np.linspace(X[feature].min(), X[feature].max(), 100)
        
        # Create scatter plot
        plt.scatter(X[feature], y, color='#2E86C1', alpha=0.5, s=50, label='Data Points')
        
        # Generate predictions for the line
        predictions = []
        for x_val in x_range:
            X_temp = X_plot.copy()
            X_temp[feature] = x_val
            pred = pipeline.predict(X_temp)
            predictions.append(np.mean(pred))
        
        # Plot fitted line
        plt.plot(x_range, predictions, color='#E74C3C', linewidth=3, label='Linear Regression')
        
        plt.xlabel(feature.upper(), fontsize=12, fontweight='bold')
        plt.ylabel('Insurance Cost ($)', fontsize=12, fontweight='bold')
        plt.title(f'Linear Regression: {feature.upper()}', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'insurance_all_features_sklearn.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  • Saved combined plot to {save_path}")
    
    # Create individual high-quality plots
    for feature in numeric_features:
        save_path = os.path.join(save_dir, f'insurance_{feature}_regression.png')
        visualize_regression_scatter(pipeline, X, y, feature, save_path)

def visualize_feature_importance(pipeline, feature_names, save_dir='plots'):
    """
    Create a bar plot of feature importance for linear regression.
    Args:
        pipeline: Trained sklearn pipeline
        feature_names: List of feature names
        save_dir: Directory to save the image
    """
    # Get feature coefficients
    coefficients = pipeline.named_steps['regressor'].coef_
    
    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps['preprocessor']
    transformed_features = (
        feature_names[:len(INSURANCE_NUMERIC)] +  # Numeric features
        [f"{feat}_{val}" for feat, vals in 
         zip(INSURANCE_CATEGORICAL, 
             preprocessor.named_transformers_['cat'].categories_) 
         for val in vals[1:]]  # Categorical features (excluding first category)
    )
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    
    # Plot horizontal bars
    y_pos = np.arange(len(coefficients))
    plt.barh(y_pos, np.abs(coefficients), color='#2E86C1', alpha=0.7)
    
    # Customize plot
    plt.yticks(y_pos, transformed_features, fontsize=10)
    plt.xlabel('Absolute Coefficient Value', fontsize=12, fontweight='bold')
    plt.title('Feature Importance in Linear Regression', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on the bars
    for i, v in enumerate(np.abs(coefficients)):
        plt.text(v, i, f' {v:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'feature_importance_sklearn.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  • Saved feature importance plot to {save_path}")

def visualize_all_features_diabetes(pipeline, X, y, save_dir='plots'):
    """
    Create scatter plots with decision boundaries for all feature pairs in the diabetes dataset.
    Args:
        pipeline: Trained sklearn pipeline
        X: DataFrame of features
        y: Series of target values
        save_dir: Directory to save the images
    """
    print_step("Creating scatter plots for diabetes features...")
    
    # Select important features for visualization
    important_features = ['Glucose', 'BMI', 'Age', 'BloodPressure']
    n_features = len(important_features)
    
    plt.figure(figsize=(15, 15))
    
    plot_idx = 1
    for i in range(n_features):
        for j in range(i+1, n_features):
            feature1, feature2 = important_features[i], important_features[j]
            
            plt.subplot(n_features-1, n_features-1, plot_idx)
            
            # Create scatter plot
            for label in [0, 1]:
                mask = y == label
                plt.scatter(X[feature1][mask], X[feature2][mask],
                          label=f'Class {label}',
                          alpha=0.6)
            
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.title(f'{feature1} vs {feature2}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'diabetes_all_features.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  • Saved combined plot to {save_path}")

def main():
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
        print("Created 'plots' directory")

    # Insurance Data (Regression)
    print_section("Insurance Cost Prediction")
    X_insurance, y_insurance = load_insurance_data('insurance.csv')
    
    X_train_ins, X_test_ins, y_train_ins, y_test_ins = train_test_split(
        X_insurance, y_insurance, test_size=0.2, random_state=42
    )
    
    # Train and evaluate Linear Regression
    lr_pipeline = train_and_evaluate_regression(
        "Linear Regression",
        LinearRegression(),
        X_train_ins, X_test_ins, y_train_ins, y_test_ins
    )
    
    # Train and evaluate Ridge Regression
    ridge_pipeline = train_and_evaluate_regression(
        "Ridge Regression",
        Ridge(alpha=1.0),
        X_train_ins, X_test_ins, y_train_ins, y_test_ins
    )
    
    # Visualize results
    print_section("Visualization - Linear Regression")
    
    # Create plots for all insurance features
    visualize_all_regression_features(lr_pipeline, X_train_ins, y_train_ins)
    
    # Create feature importance plot
    visualize_feature_importance(lr_pipeline, 
                               X_insurance.columns.tolist(),
                               save_dir='plots')
    
    # Diabetes Data (Classification)
    print_section("Diabetes Prediction")
    X_diabetes, y_diabetes = load_diabetes_data('diabetes-dataset.csv')
    
    X_train_dia, X_test_dia, y_train_dia, y_test_dia = train_test_split(
        X_diabetes, y_diabetes, test_size=0.2, random_state=42
    )
    
    # Train and evaluate Logistic Regression
    log_pipeline = train_and_evaluate_classification(
        "Logistic Regression",
        LogisticRegression(max_iter=1000),
        X_train_dia, X_test_dia, y_train_dia, y_test_dia
    )
    
    # Create confusion matrix
    y_pred_train = log_pipeline.predict(X_train_dia)
    visualize_confusion_matrix(y_train_dia, y_pred_train,
                             save_path='plots/diabetes_confusion_matrix.png')

if __name__ == "__main__":
    main()