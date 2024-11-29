## Multi-Day Trip Cluster

This repository contains a FastAPI-based project that performs clustering using a custom K-means implementation in TensorFlow. It also includes location scheduling based on business hours and proximity between place.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Folder Structure](#folder-structure)
6. [API Endpoints](#api-endpoints)
7. [Example](#example)
   - [Request](#request)
   - [Response](#response)
8. [Bonus (Optional)](#bonus-optional)

## Features

- **Clustering with TensorFlow**: Custom K-means implementation with soft penalties for outliers.
- **Location Scheduling**: Allocates time slots for visiting clustered locations within business hours.

## Prerequisites

- **Python 3.9**
- **Conda** (Optional)

## Installation

Example of setting up a virtual environment using **Conda** to manage dependencies and Python versions effectively. You can also use other virtual environments, such as venv or virtualenv.

1. Clone the repository:

   ```bash
   https://github.com/C242-DT02-Day-Trips-Optimization/Machine_Learning.git
   cd Machine_Learning
   ```

2. Create a new Conda environment:

   ```bash
   conda create --name cluster_env python=3.9
   conda activate cluster_env
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the FastAPI application, use **Uvicorn**, an ASGI server that runs the FastAPI app. The main API is served through `main.py`.

1. Run the FastAPI server:

   ```bash
   uvicorn app.main:app --reload
   ```

2. Once the server is running, navigate to `http://127.0.0.1:8000/docs` to view the interactive API documentation (Swagger UI).

## Folder Structure

The main structure of this project is:

```
fastapi-clustering/
├── app/
│   ├── clustering.py           # Customized TensorFlow K-means clustering
│   ├── evaluation.py           # Evaluation metrics (silhouette score, etc.)
│   ├── main.py                 # FastAPI initialization
│   ├── models.py               # Pydantic models
│   ├── routes.py               # API endpoints for clustering and stuff
│   ├── scheduling.py           # Scheduling logic for each clustered locations
│   ├── utils.py                # Utility functions for normalization and visualization
├── requirements.txt            # Dependencies
```

## API Endpoints

**Cluster Locations** (`POST /cluster/`):
   - Clusters locations based on coordinates and schedules visits within specified business hours.
   - **Input**: JSON object containing `Location` data points and clustering parameters.
   - **Output**: Grouped clusters with schedules and unvisitable locations.

## Example 

### Request

Use this example JSON to test the `/cluster/` endpoint:

```json
{
    "points": [
        {
            "name": "Uluwatu Temple",
            "coordinates": [-8.831325, 115.088114],
            "opening_hours": "08:00",
            "closing_hours": "18:00",
            "duration": 2
        },
        {
            "name": "Tanah Lot",
            "coordinates": [-8.621069, 115.086853],
            "opening_hours": "09:00",
            "closing_hours": "19:00",
            "duration": 2
        },
        {
            "name": "Tegallalang Rice Terraces",
            "coordinates": [-8.432983, 115.279935],
            "opening_hours": "07:00",
            "closing_hours": "17:00",
            "duration": 3
        },
        {
            "name": "Ubud Monkey Forest",
            "coordinates": [-8.518644, 115.258481],
            "opening_hours": "08:30",
            "closing_hours": "18:00",
            "duration": 2
        },
        {
            "name": "Kuta Beach",
            "coordinates": [-8.717879, 115.169208],
            "opening_hours": "06:00",
            "closing_hours": "20:00",
            "duration": 3
        },
        {
            "name": "Mount Batur",
            "coordinates": [-8.242036, 115.375289],
            "opening_hours": "03:00",
            "closing_hours": "15:00",
            "duration": 4
        },
        {
            "name": "Sanur Beach",
            "coordinates": [-8.688614, 115.261849],
            "opening_hours": "06:00",
            "closing_hours": "18:00",
            "duration": 2
        },
        {
            "name": "Bali Safari and Marine Park",
            "coordinates": [-8.606277, 115.319167],
            "opening_hours": "09:00",
            "closing_hours": "17:00",
            "duration": 4
        },
        {
            "name": "Pura Besakih",
            "coordinates": [-8.384272, 115.451038],
            "opening_hours": "07:00",
            "closing_hours": "18:00",
            "duration": 3
        },
        {
            "name": "Nusa Dua Beach",
            "coordinates": [-8.801580, 115.226798],
            "opening_hours": "07:00",
            "closing_hours": "19:00",
            "duration": 2
        }
    ],
    "num_clusters": 3,
    "daily_start_time": "07:00",
    "daily_end_time": "21:00"
}
```

### Response

```json
{
  "grouped_clusters": [
    {
      "cluster": 0,
      "schedule": [
        {
          "name": "Sanur Beach",
          "start_time": "07:00",
          "end_time": "09:00"
        },
        {
          "name": "Bali Safari and Marine Park",
          "start_time": "09:00",
          "end_time": "13:00"
        },
        {
          "name": "Ubud Monkey Forest",
          "start_time": "13:00",
          "end_time": "15:00"
        }
      ]
    },
    {
      "cluster": 1,
      "schedule": [
        {
          "name": "Kuta Beach",
          "start_time": "07:00",
          "end_time": "10:00"
        },
        {
          "name": "Nusa Dua Beach",
          "start_time": "10:00",
          "end_time": "12:00"
        },
        {
          "name": "Uluwatu Temple",
          "start_time": "12:00",
          "end_time": "14:00"
        },
        {
          "name": "Tanah Lot",
          "start_time": "14:00",
          "end_time": "16:00"
        }
      ]
    },
    {
      "cluster": 2,
      "schedule": [
        {
          "name": "Mount Batur",
          "start_time": "07:00",
          "end_time": "11:00"
        },
        {
          "name": "Pura Besakih",
          "start_time": "11:00",
          "end_time": "14:00"
        },
        {
          "name": "Tegallalang Rice Terraces",
          "start_time": "14:00",
          "end_time": "17:00"
        }
      ]
    }
  ],
  "final_unvisitable": []
}
```

# Bonus (Optional)

**Uncomment** the visualization and response in routes.py to gain a better understanding of the clustering and the routing (scheduling) response (or you could check out the static folder)

<img src="https://github.com/Gumm11/Multi-Day-Cluster/blob/main/static/cluster_plot.png" width="600" />
<img src="https://github.com/Gumm11/Multi-Day-Cluster/blob/main/static/routing_plot.png" width="600" />
<img src="https://github.com/Gumm11/Multi-Day-Cluster/blob/main/static/schedule_table_cluster_0.png" width="600" />
<img src="https://github.com/Gumm11/Multi-Day-Cluster/blob/main/static/schedule_table_cluster_1.png" width="600" />
<img src="https://github.com/Gumm11/Multi-Day-Cluster/blob/main/static/schedule_table_cluster_2.png" width="600" />
