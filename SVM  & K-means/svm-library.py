import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    try:
        # Load the dataset
        print("Loading diabetes dataset...")
        df = pd.read_csv('diabetes-dataset.csv')
        
        # Preprocess the data
        print("\nPreprocessing data...")
        # Replace zeros with NaN in specific columns
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[zero_cols] = df[zero_cols].replace(0, np.nan)
        
        # Fill missing values with median
        df.fillna(df.median(), inplace=True)
        
        # Convert outcome to class 1 and 2
        df['Outcome'] = df['Outcome'].replace({0: 1, 1: 2})
        
        # Split features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split the data
        print("\nSplitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM model
        print("\nTraining SVM model...")
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = svm_model.predict(X_test_scaled)
        
        # Calculate class distribution
        total_points = len(y_pred)
        class_1_count = sum(1 for pred in y_pred if pred == 1)
        class_2_count = sum(1 for pred in y_pred if pred == 2)
        class_1_percentage = (class_1_count / total_points) * 100
        class_2_percentage = (class_2_count / total_points) * 100
        
        # Print results
        print("\nPrediction Distribution:")
        print(f"Total points: {total_points}")
        print(f"Class 1: {class_1_count} points ({class_1_percentage:.2f}%)")
        print(f"Class 2: {class_2_count} points ({class_2_percentage:.2f}%)")
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.2f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2']))
        
        # Create and save confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 1', 'Class 2'],
                   yticklabels=['Class 1', 'Class 2'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.savefig('svm_confusion_matrix.png')
        plt.close()
        
        print("\nAnalysis complete! Check 'svm_confusion_matrix.png' for visualization.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 