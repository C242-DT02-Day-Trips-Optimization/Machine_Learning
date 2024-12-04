from geopy.distance import geodesic
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datetime import datetime
import googlemaps
import math
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the Google Maps client with the actual API key from the environment variable
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key is not set in environment variables.")

gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

def get_travel_time(start_coords, end_coords, distance_threshold=1.0):
    """
    Get travel time between two locations using Google Maps API.
    :param start_coords: Tuple (lat, lon) for the starting location.
    :param end_coords: Tuple (lat, lon) for the ending location.
    :param distance_threshold: The threshold distance in kilometers to determine walking or driving mode.
    :return: Travel time in minutes (rounded up) and the mode used (walking/driving).
    """
    # Calculate the distance between start and end locations using geodesic
    distance = geodesic(start_coords, end_coords).kilometers

    # Determine the mode based on distance
    mode = "walking" if distance <= distance_threshold else "driving"
    
    # Request directions between start and end using the selected mode
    directions = gmaps.directions(
        origin=start_coords,
        destination=end_coords,
        mode=mode,
        departure_time=datetime.now()
    )
    
    # Extract travel time (duration) in minutes
    if directions:
        duration = directions[0]['legs'][0]['duration']['value'] / 60  # duration in minutes
        rounded_duration = math.ceil(duration)  # Round up to the nearest minute
        return rounded_duration, mode  # Return both rounded duration and mode
    else:
        # If no directions are found, return a default value (or handle as needed)
        return 0, mode

# Helper function to calculate the total available time in minutes
def calculate_total_time_in_minutes(daily_start_time, daily_end_time):
    """
    Calculate total available time in minutes based on start and end times.
    """
    start_time = datetime.strptime(daily_start_time, "%H:%M")
    end_time = datetime.strptime(daily_end_time, "%H:%M")
    total_time = (end_time - start_time).seconds / 60  # Convert seconds to minutes
    return total_time

# Function to calculate the average duration for each location in the cluster
def calculate_average_duration(total_time, num_locations):
    """
    Calculate the average time allocated for each location within a cluster.
    """
    if num_locations == 0:
        return 0
    return total_time / num_locations

# Function to handle unvisitable locations and fit them into other clusters based on proximity
def handle_unvisitable(locations, clusters):
    new_unvisitable = []
    for location in locations:
        fit = False
        for cluster_id, cluster_schedule in clusters.items():
            result = schedule_single_location(location, cluster_schedule["schedule"])
            if result:
                cluster_schedule["schedule"].append(result)
                cluster_schedule["schedule"].sort(key=lambda x: x["proximity_to_next"])
                fit = True
                break
        if not fit:
            new_unvisitable.append(location)
    return {"clusters": clusters, "unvisitable": new_unvisitable}

# Function to schedule locations within a cluster based purely on proximity
def schedule_cluster_with_proximity(cluster, daily_start_time_str, daily_end_time_str):
    """
    Schedule locations within a cluster, first considering travel time, and then calculating the avg_duration.
    The travel time mode is determined by the distance between locations (walking vs. driving).
    """
    schedule = []
    unvisitable = []

    daily_start_time = datetime.strptime(daily_start_time_str, "%H:%M")
    daily_end_time = datetime.strptime(daily_end_time_str, "%H:%M")

    # Calculate the total time available for scheduling
    total_available_time = (daily_end_time - daily_start_time).seconds // 60  # In minutes
    total_locations = len(cluster)

    # Calculate total travel time
    total_travel_time = 0
    for i in range(len(cluster) - 1):
        travel_time, mode = get_travel_time(cluster[i].coordinates, cluster[i + 1].coordinates)
        total_travel_time += travel_time

    # Subtract total travel time from available time to calculate time for activities
    time_for_activities = total_available_time - total_travel_time

    # Calculate the average duration per location (if there are locations)
    if total_locations > 0:
        avg_duration = time_for_activities // total_locations
    else:
        avg_duration = 0

    # Start scheduling with the first location
    if cluster:
        first_location = cluster[0]
        schedule.append(create_schedule_entry(first_location, avg_duration=avg_duration))
        cluster.remove(first_location)

    while cluster:
        # Find the nearest location based on proximity
        last_location = schedule[-1]
        next_location = min(cluster, key=lambda loc: geodesic(last_location["coordinates"], loc.coordinates).kilometers)
        cluster.remove(next_location)

        # Schedule the location with the calculated avg_duration
        travel_time, mode = get_travel_time(last_location["coordinates"], next_location.coordinates)
        result = create_schedule_entry(next_location, avg_duration=avg_duration, travel_time=travel_time, mode=mode)
        if result:
            schedule.append(result)
        else:
            unvisitable.append(next_location)

    # Add proximity details to each scheduled location
    for i in range(len(schedule) - 1):
        distance_to_next = geodesic(schedule[i]["coordinates"], schedule[i + 1]["coordinates"]).kilometers
        schedule[i]["proximity_to_next"] = f"{distance_to_next:.2f} km"

    if schedule:
        schedule[-1]["proximity_to_next"] = "N/A"  # Last location has no next location

    return {"schedule": schedule, "unvisitable": unvisitable}

# Function to schedule a single location based on proximity
def schedule_single_location(location, current_schedule, avg_duration=None):
    """
    Schedule a single location within a cluster, purely based on proximity and average duration.
    
    Parameters:
        location (dict): The location to schedule, with details such as coordinates.
        current_schedule (list): Current schedule of the cluster.
        avg_duration (int): The average duration for the cluster to be used for this location.
        
    Returns:
        dict or None: The scheduled entry or None if the location cannot be scheduled.
    """
    if not current_schedule:
        # No locations scheduled yet, schedule the first one
        return create_schedule_entry(location, avg_duration=avg_duration)

    last_location = current_schedule[-1]
    distance_to_last = geodesic(last_location["coordinates"], location.coordinates).kilometers
    return create_schedule_entry(location, proximity_to_last=distance_to_last, avg_duration=avg_duration)

# Helper function to create a schedule entry with proximity information
def create_schedule_entry(location, proximity_to_last=None, avg_duration=None, travel_time=None, mode=None):
    """
    Return schedule entry with the location's details, proximity to last location,
    average duration if provided, and travel time and mode.
    """
    entry = {
        "name": location.name,
        "coordinates": location.coordinates,
    }
    
    if proximity_to_last is not None:
        entry["proximity_to_last"] = f"{proximity_to_last:.2f} km"
    
    if avg_duration is not None:
        entry["avg_duration"] = avg_duration
    
    if travel_time is not None:
        entry["travel_time"] = travel_time  # Travel time in minutes
    
    if mode is not None:
        entry["mode"] = mode  # Travel mode: 'driving' or 'walking'
    
    return entry

# Function to run scheduling in parallel for multiple clusters
def parallel_schedule_clusters(clusters, daily_start_time, daily_end_time, num_threads=4):
    """
    Schedules multiple clusters in parallel using multithreading.
    
    Args:
        clusters (dict): Dictionary of clusters where key is cluster ID and value is list of locations.
        daily_start_time (datetime): The start time of the day.
        daily_end_time (datetime): The end time of the day.
        num_threads (int): Number of threads to use for parallel processing.
        
    Returns:
        dict: Dictionary with scheduled clusters and unvisitable locations.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            cluster_id: executor.submit(
                schedule_cluster_with_proximity, 
                cluster_locations, 
                daily_start_time, 
                daily_end_time
            )
            for cluster_id, cluster_locations in clusters.items()
        }
        for cluster_id, future in futures.items():
            cluster_data = future.result()
            
            # Calculate the average duration for the cluster
            schedule = cluster_data["schedule"]
            if schedule:
                total_duration = sum(location["avg_duration"] for location in schedule)
                avg_duration = total_duration // len(schedule)
            else:
                avg_duration = 0

            # Add the avg_duration to the response for each cluster
            results[cluster_id] = {
                "schedule": schedule,
                "avg_duration": avg_duration,
                "unvisitable": cluster_data["unvisitable"]
            }

    return results

# Function to run multiple iterations of scheduling and find the best schedule
def iterative_schedule_cluster(cluster, max_iterations=20):
    """
    Find the best schedule for a single cluster from multiple iterations.
    
    Return a dict of best schedule and any remaining unvisitable locations.
    """
    best_schedule = None
    best_unvisitable = None
    best_score = float("inf")

    for _ in range(max_iterations):
        result = schedule_cluster_with_proximity(cluster[:])
        unvisitable_count = len(result["unvisitable"])
        schedule_length = len(result["schedule"])
        score = unvisitable_count * 1000 - schedule_length

        if score < best_score:
            best_score = score
            best_schedule = result["schedule"]
            best_unvisitable = result["unvisitable"]

        # Stop early if no unvisitable locations
        if unvisitable_count == 0:
            break

    return {"schedule": best_schedule, "unvisitable": best_unvisitable}