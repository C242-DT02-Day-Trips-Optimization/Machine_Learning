from fastapi import APIRouter, HTTPException
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# import time
import pandas as pd
import joblib
from app.models import ClusteringInput
from app.clustering import tensorflow_kmeans
from app.scheduling import (
    schedule_cluster_with_proximity,
    handle_unvisitable,
    parallel_schedule_clusters
)
# from app.utils import visualize_clusters, visualize_routing#, generate_schedule_table
from app.evaluation import (
    compute_silhouette_score,
    # compute_davies_bouldin_index,
    # compute_intra_cluster_distance,
)
from concurrent.futures import ThreadPoolExecutor
import sys

sys.dont_write_bytecode = True

# Initialize the API router for clustering-related endpoints
clustering_router = APIRouter()

@clustering_router.post("/cluster/", summary="Cluster Locations and Generate Schedules")
def cluster_data(data: ClusteringInput): 
    """
    Endpoint to perform clustering on location data and generate schedules
    - Takes in a set of location points and performs clustering based on the specified number of clusters
    - Schedules the visits purely based on proximity, handles any unvisitable locations, and returns the clustering results
    """
    coordinates = np.array([loc.coordinates for loc in data.points])
    names = [loc.name for loc in data.points]
    locations = {loc.name: loc for loc in data.points}
    num_clusters = data.num_clusters
    daily_start_time = data.daily_start_time  # Assuming these are in HH:MM format
    daily_end_time = data.daily_end_time

    # Input validation
    if num_clusters < 1:
        raise HTTPException(status_code=400, detail="Number of clusters must be at least 1")

    if len(coordinates) < num_clusters:
        raise HTTPException(status_code=400, detail="Number of clusters cannot exceed number of points")

    # Normalize coordinates to scale features between 0 and 1
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(coordinates)

    if num_clusters == 1:
        best_labels = np.zeros(len(coordinates), dtype=int)  # All points belong to cluster 0
        best_centroids = normalized_data.mean(axis=0).reshape(1, -1)  # Centroid = mean of all points
        best_clusters = {0: list(locations.values())}  # All locations in cluster 0
    else:
        # Perform parallel clustering to find the best cluster
        best_clusters, best_labels, best_centroids, best_metrics = parallel_find_best_clusters(
            normalized_data, locations, num_clusters, num_iterations=8
        )
        
    # # Visualize the clustering results
    # cluster_plot_path = "static/cluster_plot.png"
    # visualize_clusters(
    #     data=normalized_data,
    #     labels=best_labels,
    #     centroids=best_centroids,
    #     output_path=cluster_plot_path,
    # )

    # Schedule the clustered locations purely based on proximity
    grouped_clusters = parallel_schedule_clusters(best_clusters, daily_start_time, daily_end_time)

    # Handle unvisitable locations
    unvisitable_locations = [
        loc for cluster in grouped_clusters.values() for loc in cluster["unvisitable"]
    ]

    adjusted_result = handle_unvisitable(unvisitable_locations, grouped_clusters)
    grouped_clusters = adjusted_result["clusters"]
    final_unvisitable = adjusted_result["unvisitable"]
    
    # # Visualize routing based on the scheduled clusters
    # routing_plot_path = "static/routing_plot.png"
    # visualize_routing(
    #     grouped_clusters=grouped_clusters,
    #     unvisitable=final_unvisitable,
    #     output_path=routing_plot_path
    # )

    # Compile final response
    response = {
        "grouped_clusters": [
            {
                "cluster": cluster_id,
                "avg_duration": cluster_data["avg_duration"],  
                "schedule": [
                    {
                        "name": loc["name"],
                        "avg_duration": loc.get("avg_duration"),
                        "travel_time": loc.get("travel_time"),
                        "mode": loc.get("mode"),
                    }
                    for loc in cluster_data["schedule"]
                ],
            }
            for cluster_id, cluster_data in grouped_clusters.items()
        ],
        "final_unvisitable": [
            {"name": loc.name, "reason": "Too many destinations for one day"}  # Update reason accordingly
            for loc in final_unvisitable
        ],
    }

    return response

# Do we need this multithreading? Or is it too much? (For clustering)
def parallel_find_best_clusters(normalized_data, locations, num_clusters, num_iterations):
    best_clusters = None
    best_labels = None
    best_centroids = None
    best_metrics = None
    best_composite_score = float("-inf")

    def cluster_and_evaluate(_):
        """
        Perform a single clustering iteration and compute evaluation metrics.
        """
        centroids, labels = tensorflow_kmeans(normalized_data, num_clusters)
        silhouette = compute_silhouette_score(normalized_data, labels)
        cluster_balance_score = compute_cluster_balance_score(labels, num_clusters)
        return centroids, labels, silhouette, cluster_balance_score

    # Run clustering iterations in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cluster_and_evaluate, range(num_iterations)))

    # Select the best cluster configuration based on composite score
    for centroids, labels, silhouette, cluster_balance in results:
        composite_score = (0.7 * silhouette) + (0.3 * (1 - cluster_balance))  # Combine metrics
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_labels = labels
            best_centroids = centroids
            best_metrics = {
                "silhouette_score": silhouette,
                "cluster_balance_score": cluster_balance,
            }
            best_clusters = {i: [] for i in range(num_clusters)}
            for name, cluster_id in zip(locations.keys(), labels):
                best_clusters[int(cluster_id)].append(locations[name])

    return best_clusters, best_labels, best_centroids, best_metrics

def compute_cluster_balance_score(labels, num_clusters):
    """
    Compute a score for cluster balance based on the variance of cluster sizes.
    Explanation: 
        - Balanced clustering has clusters with similar sizes = lower variance.
        - Balance score normalized by the mean cluster size for comparability.

    Returns float of normalized variance of cluster sizes (lower values indicate better balance)
    """
    cluster_counts = [np.sum(labels == i) for i in range(num_clusters)]
    balance_score = np.var(cluster_counts) / np.mean(cluster_counts)  # Normalize by mean
    return balance_score

# ------------------------------------- HDBSCAN AREA -------------------------------------

# Load pre-trained HDBSCAN model
def load_model(province: str):
    if province.lower() == ('jawa timur' or 'east java'):
        model_path = 'model/jatim.pkl'  
    elif province.lower() == ('jawa barat' or 'west java'):
        model_path = 'model/jabar.pkl'  
    elif province.lower() == ('jawa tengah' or 'central java'):
        model_path = 'model/jateng.pkl'  
    elif province.lower() == ('jakarta' or 'dki jakarta'):
        model_path = 'model/jakarta.pkl'  
    elif province.lower() == 'bali':
        model_path = 'model/bali.pkl'  
    else:
        raise ValueError("Invalid province")
    
    # Load the model
    return joblib.load(model_path)

# Endpoint for generating a recommended schedule using the pre-trained model (HDBSCAN version)
@clustering_router.post("/recommend/") 
async def generate_trip_schedule(input_data: ClusteringInput):
    
    # Load the appropriate HDBSCAN model based on the province
    clusterer = load_model(input_data.province)

    # Extracting coordinates from user input
    user_coords = [(loc.coordinates[0], loc.coordinates[1]) for loc in input_data.points]

    # Convert coordinates to radians (HDBSCAN requires radians for clustering)
    user_coords_rad = np.radians(user_coords)

    # Predict clusters using the pre-trained HDBSCAN model
    cluster_labels = clusterer.fit_predict(user_coords_rad)

    # Prepare the response in the specified format
    grouped_clusters = {}
    for cluster_id in set(cluster_labels):
        cluster_schedule = [input_data.points[i] for i in range(len(cluster_labels)) if cluster_labels[i] == cluster_id]
        grouped_clusters[cluster_id] = cluster_schedule

    # Now apply the scheduling functions to the clusters
    daily_start_time = input_data.daily_start_time
    daily_end_time = input_data.daily_end_time

    # Schedule clusters based on proximity
    scheduled_clusters = parallel_schedule_clusters(grouped_clusters, daily_start_time, daily_end_time)

    # Handle unvisitable locations
    unvisitable_locations = [
        loc for cluster_data in scheduled_clusters.values() for loc in cluster_data["unvisitable"]
    ]
    adjusted_result = handle_unvisitable(unvisitable_locations, scheduled_clusters)
    scheduled_clusters = adjusted_result["clusters"]
    final_unvisitable = adjusted_result["unvisitable"]

    # # Visualize routing based on the scheduled clusters
    # routing_plot_path = "static/routing_plot.png"
    # visualize_routing(
    #     grouped_clusters=scheduled_clusters,
    #     unvisitable=final_unvisitable,
    #     output_path=routing_plot_path
    # )
    
    recommended_days = len(grouped_clusters)

    # Compile final response
    response = {
        "grouped_clusters": [
            {
                "cluster": cluster_id,
                "schedule": [
                    {
                        "name": loc["name"],
                        "avg_duration": loc.get("avg_duration"),
                        "travel_time": loc.get("travel_time"),
                        "mode": loc.get("mode"),
                    }
                    for loc in cluster_data["schedule"]
                ],
                "avg_duration": cluster_data["avg_duration"],
            }
            for cluster_id, cluster_data in scheduled_clusters.items()
        ],
        "final_unvisitable": [
            {"name": loc["name"], "reason": "Too many destinations for one day"}
            for loc in final_unvisitable
        ],
        "recommended_days": recommended_days 
    }

    response = convert_numpy_objects(response)

    return response

def convert_numpy_objects(obj):
    """Convert all numpy types to native Python types recursively."""
    if isinstance(obj, np.int64):
        return int(obj)  # Convert numpy int64 to native int
    elif isinstance(obj, np.float64):
        return float(obj)  # Convert numpy float64 to native float
    elif isinstance(obj, dict):
        return {key: convert_numpy_objects(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_objects(item) for item in obj]
    return obj  # Return the object as is if it's not a numpy type
