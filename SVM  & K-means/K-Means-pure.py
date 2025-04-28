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

def euclidean_distance(a, b):
    """Calculate Euclidean distance between two points."""
    try:
        if len(a) != len(b):
            return float('inf')
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    except Exception:
        return float('inf')

def initialize_centroids(data, k):
    """Initialize k centroids using k-means++ initialization."""
    try:
        if not data or len(data) < k:
            return None
            
        centroids = [random.choice(data)[:-1]]  # First centroid is random
        for _ in range(1, k):
            distances = []
            for point in data:
                min_dist = min(euclidean_distance(point[:-1], c) for c in centroids)
                distances.append(min_dist)
            
            # Choose next centroid with probability proportional to distance squared
            total = sum(d * d for d in distances)
            r = random.random() * total
            cumulative = 0
            for i, d in enumerate(distances):
                cumulative += d * d
                if cumulative >= r:
                    centroids.append(data[i][:-1])
                    break
        
        return centroids
    except Exception as e:
        print(f"Error initializing centroids: {str(e)}")
        return None

def assign_clusters_batch(data, centroids):
    """Assign clusters using batch processing."""
    try:
        if not data or not centroids:
            return None
            
        clusters = []
        for point in data:
            distances = [euclidean_distance(point[:-1], centroid) for centroid in centroids]
            cluster = distances.index(min(distances))
            clusters.append(cluster)
        return clusters
    except Exception as e:
        print(f"Error in batch assignment: {str(e)}")
        return None

def assign_clusters_minibatch(data, centroids, batch_size=32):
    """Assign clusters using mini-batch processing."""
    try:
        if not data or not centroids:
            return None
            
        clusters = [0] * len(data)
        indices = list(range(len(data)))
        random.shuffle(indices)
        
        # Process data in batches
        for i in range(0, len(data), batch_size):
            batch = indices[i:i + batch_size]
            for idx in batch:
                point = data[idx]
                # Calculate distances to all centroids
                distances = [euclidean_distance(point[:-1], centroid) for centroid in centroids]
                # Assign to closest centroid
                clusters[idx] = distances.index(min(distances))
        
        return clusters
    except Exception as e:
        print(f"Error in mini-batch assignment: {str(e)}")
        return None

def assign_clusters_stochastic(data, centroids):
    """Assign clusters using stochastic processing."""
    try:
        if not data or not centroids:
            return None
            
        clusters = [0] * len(data)
        indices = list(range(len(data)))
        random.shuffle(indices)
        
        for idx in indices:
            point = data[idx]
            distances = [euclidean_distance(point[:-1], centroid) for centroid in centroids]
            clusters[idx] = distances.index(min(distances))
        
        return clusters
    except Exception as e:
        print(f"Error in stochastic assignment: {str(e)}")
        return None

def update_centroids_batch(data, clusters, k):
    """Update centroids using batch processing."""
    try:
        if not data or not clusters or len(data) != len(clusters):
            return None
            
        new_centroids = []
        for i in range(k):
            cluster_points = [point[:-1] for point, cluster in zip(data, clusters) if cluster == i]
            if cluster_points:
                new_centroid = [sum(x) / len(cluster_points) for x in zip(*cluster_points)]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.choice(data)[:-1])
        return new_centroids
    except Exception as e:
        print(f"Error in batch centroid update: {str(e)}")
        return None

def update_centroids_minibatch(data, clusters, k, batch_size=32):
    """Update centroids using mini-batch processing."""
    try:
        if not data or not clusters or len(data) != len(clusters):
            return None
            
        # Initialize centroids with zeros
        new_centroids = [[0.0] * (len(data[0]) - 1) for _ in range(k)]
        counts = [0] * k
        
        # Process data in batches
        indices = list(range(len(data)))
        random.shuffle(indices)
        
        for i in range(0, len(data), batch_size):
            batch = indices[i:i + batch_size]
            for idx in batch:
                cluster = clusters[idx]
                point = data[idx][:-1]
                # Update centroid sums
                for j in range(len(point)):
                    new_centroids[cluster][j] += point[j]
                counts[cluster] += 1
        
        # Normalize centroids
        for i in range(k):
            if counts[i] > 0:
                new_centroids[i] = [x / counts[i] for x in new_centroids[i]]
            else:
                # If a cluster has no points, reinitialize its centroid
                new_centroids[i] = random.choice(data)[:-1]
        
        return new_centroids
    except Exception as e:
        print(f"Error in mini-batch centroid update: {str(e)}")
        return None

def update_centroids_stochastic(data, clusters, k):
    """Update centroids using stochastic processing."""
    try:
        if not data or not clusters or len(data) != len(clusters):
            return None
            
        new_centroids = [[0.0] * (len(data[0]) - 1) for _ in range(k)]
        counts = [0] * k
        
        indices = list(range(len(data)))
        random.shuffle(indices)
        
        for idx in indices:
            cluster = clusters[idx]
            point = data[idx][:-1]
            for j in range(len(point)):
                new_centroids[cluster][j] += point[j]
            counts[cluster] += 1
        
        for i in range(k):
            if counts[i] > 0:
                new_centroids[i] = [x / counts[i] for x in new_centroids[i]]
            else:
                new_centroids[i] = random.choice(data)[:-1]
        
        return new_centroids
    except Exception as e:
        print(f"Error in stochastic centroid update: {str(e)}")
        return None

def kmeans_batch(data, k, max_iterations=100):
    """Perform K-means clustering using batch processing."""
    try:
        if not data or len(data) < k:
            return None, None
            
        centroids = initialize_centroids(data, k)
        if not centroids:
            return None, None
            
        for _ in range(max_iterations):
            clusters = assign_clusters_batch(data, centroids)
            if not clusters:
                return None, None
                
            new_centroids = update_centroids_batch(data, clusters, k)
            if not new_centroids:
                return None, None
                
            if all(all(x == y for x, y in zip(c1, c2)) for c1, c2 in zip(centroids, new_centroids)):
                break
            centroids = new_centroids
        
        return clusters, centroids
    except Exception as e:
        print(f"Error in batch K-means: {str(e)}")
        return None, None

def kmeans_minibatch(data, k, max_iterations=100, batch_size=32):
    """Perform K-means clustering using mini-batch processing."""
    try:
        if not data or len(data) < k:
            return None, None
            
        # Initialize centroids using k-means++
        centroids = initialize_centroids(data, k)
        if not centroids:
            return None, None
            
        # Track previous centroids for convergence check
        prev_centroids = None
        
        for iteration in range(max_iterations):
            # Assign clusters
            clusters = assign_clusters_minibatch(data, centroids, batch_size)
            if not clusters:
                return None, None
                
            # Update centroids
            new_centroids = update_centroids_minibatch(data, clusters, k, batch_size)
            if not new_centroids:
                return None, None
            
            # Check for convergence
            if prev_centroids is not None:
                # Calculate centroid movement
                movement = sum(
                    euclidean_distance(c1, c2)
                    for c1, c2 in zip(centroids, new_centroids)
                )
                if movement < 1e-6:  # Small threshold for convergence
                    break
            
            prev_centroids = centroids
            centroids = new_centroids
            
            # Print progress
            if iteration % 10 == 0:
                cluster_counts = [0] * k
                for cluster in clusters:
                    cluster_counts[cluster] += 1
                print(f"Iteration {iteration}: Cluster sizes: {cluster_counts}")
        
        return clusters, centroids
    except Exception as e:
        print(f"Error in mini-batch K-means: {str(e)}")
        return None, None

def kmeans_stochastic(data, k, max_iterations=100):
    """Perform K-means clustering using stochastic processing."""
    try:
        if not data or len(data) < k:
            return None, None
            
        centroids = initialize_centroids(data, k)
        if not centroids:
            return None, None
            
        for _ in range(max_iterations):
            clusters = assign_clusters_stochastic(data, centroids)
            if not clusters:
                return None, None
                
            new_centroids = update_centroids_stochastic(data, clusters, k)
            if not new_centroids:
                return None, None
                
            if all(all(x == y for x, y in zip(c1, c2)) for c1, c2 in zip(centroids, new_centroids)):
                break
            centroids = new_centroids
        
        return clusters, centroids
    except Exception as e:
        print(f"Error in stochastic K-means: {str(e)}")
        return None, None

def main():
    try:
        print("Loading diabetes dataset...")
        data, headers = load_data('diabetes-dataset.csv')
        if data is None:
            return
        
        print("\nPreprocessing data...")
        data = preprocess_data(data)
        if data is None:
            return
        
        k = 2  # Fixed number of clusters
        
        # Train with different methods
        methods = {
            "Batch": kmeans_batch,
            "Mini-batch": kmeans_minibatch,
            "Stochastic": kmeans_stochastic
        }
        
        for method_name, kmeans_func in methods.items():
            print(f"\nPerforming K-means clustering using {method_name}...")
            clusters, centroids = kmeans_func(data, k)
            
            if clusters and centroids:
                # Calculate cluster distribution
                cluster_counts = defaultdict(int)
                for cluster in clusters:
                    cluster_counts[cluster] += 1
                
                print(f"\nCluster Distribution ({method_name}):")
                for cluster, count in sorted(cluster_counts.items()):
                    percentage = (count / len(clusters)) * 100
                    print(f"Cluster {cluster}: {count} points ({percentage:.2f}%)")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 