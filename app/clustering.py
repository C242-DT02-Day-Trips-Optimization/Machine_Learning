import tensorflow as tf

def tensorflow_kmeans(data, num_clusters, num_iterations=50, penalty_threshold=0.3, penalty_factor=0.5, tol=1e-4):
    """
    Custom K-means clustering implementation using TensorFlow with soft penalties for distant points.

    This function performs K-means clustering, enhanced with a soft penalty mechanism to reduce 
    the influence of outliers or distant points on centroid updates. The algorithm is useful for 
    datasets where clusters are unevenly distributed or outliers are present.

    Parameters:
        data (numpy.ndarray or tf.Tensor): The input dataset, where each row represents a data point.
        num_clusters (int): The number of clusters to form.
        num_iterations (int, optional): The maximum number of iterations to perform. Default is 100.
        penalty_threshold (float, optional): Distance threshold beyond which soft penalties are applied. Default is 0.3.
        penalty_factor (float, optional): Factor controlling the penalty strength for distant points. Default is 0.5.

    Returns:
        tuple:
            - centroids (numpy.ndarray): Final centroid positions after clustering.
            - cluster_assignments (numpy.ndarray): Cluster indices assigned to each data point.
    """
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    initial_centroids = tf.slice(tf.random.shuffle(data), [0, 0], [num_clusters, -1])
    centroids = tf.Variable(initial_centroids)
    prev_centroids = tf.Variable(tf.zeros_like(centroids))  # Store centroids from the previous iteration

    for _ in range(num_iterations):
        # Compute distances from points to centroids (Euclidean distance)
        distances = tf.norm(data[:, None] - centroids, axis=2)

        # Assign each data point to the cluster of the nearest centroid
        cluster_assignments = tf.argmin(distances, axis=1)

        # Update centroids with soft penalties
        new_centroids = []
        for c in range(num_clusters):
            # Get points in the current cluster
            cluster_points = tf.gather(data, tf.where(cluster_assignments == c)[:, 0])
            if cluster_points.shape[0] == 0:  # Handle empty clusters
                new_centroids.append(centroids[c])
                continue

            # Calculate distances of points to the current centroid
            cluster_distances = tf.norm(cluster_points - centroids[c], axis=1)

            # Apply penalty: Weight decreases for distant points
            weights = tf.maximum(1.0 - penalty_factor * tf.maximum(cluster_distances - penalty_threshold, 0), 0.01)
            weighted_sum = tf.reduce_sum(cluster_points * weights[:, None], axis=0)
            weight_sum = tf.reduce_sum(weights)

            # Calculate new centroid
            new_centroid = weighted_sum / weight_sum
            new_centroids.append(new_centroid)

        # Update centroids
        centroids.assign(tf.stack(new_centroids))

        # Check for early stopping
        centroid_shift = tf.reduce_max(tf.norm(centroids - prev_centroids, axis=1))  # Largest centroid movement
        if centroid_shift.numpy() < tol:
            break

        # Update previous centroids
        prev_centroids.assign(centroids)

    return centroids.numpy(), cluster_assignments.numpy()