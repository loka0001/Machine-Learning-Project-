import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load the diabetes dataset
print("Loading diabetes dataset...")
df = pd.read_csv('diabetes-dataset.csv')

# Data Preprocessing
print("\nPreprocessing data...")
# Replace zeros with NaN for columns where zero is not a valid value
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_columns] = df[zero_columns].replace(0, np.nan)

# Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[zero_columns] = imputer.fit_transform(df[zero_columns])

# Convert Outcome to class 1 or 2
df['Outcome'] = df['Outcome'].replace({0: 1, 1: 2})

# Select features for clustering
features = ['Glucose', 'BMI', 'Age', 'BloodPressure']
X = df[features].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===========================================
# K-means Clustering
# ===========================================
print("\nPerforming K-means clustering...")

# Determine optimal number of clusters
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Choose optimal k based on silhouette score
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k}")

# Perform K-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = clusters

# Visualize clusters
print("\nGenerating cluster visualizations...")
plt.figure(figsize=(12, 10))
sns.pairplot(df, vars=features, hue='Cluster', palette='husl')
plt.suptitle('Pair Plot of Features by Cluster', y=1.02)
plt.savefig('kmeans_pairplot_lib.png', bbox_inches='tight')
plt.close()

# ===========================================
# Diabetes Prediction
# ===========================================
print("\nTraining diabetes prediction model...")

# Prepare features and target
X_pred = df.drop(['Outcome', 'Cluster'], axis=1)
y = df['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_pred, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate class counts and percentages
total_points = len(y_pred)
class_1_count = sum(1 for pred in y_pred if pred == 1)
class_2_count = sum(1 for pred in y_pred if pred == 2)
class_1_percentage = (class_1_count / total_points) * 100
class_2_percentage = (class_2_count / total_points) * 100

print("\nPrediction Distribution:")
print(f"Total points: {total_points}")
print(f"Class 1: {class_1_count} points ({class_1_percentage:.2f}%)")
print(f"Class 2: {class_2_count} points ({class_2_percentage:.2f}%)")

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks([0, 1], ['Class 1', 'Class 2'])
plt.yticks([0, 1], ['Class 1', 'Class 2'])
plt.savefig('confusion_matrix_lib.png')
plt.close()

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_pred.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance_lib.png')
plt.close()

print("\nAnalysis complete! Check the generated plots in your directory.") 