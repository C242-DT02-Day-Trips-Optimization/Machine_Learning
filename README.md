## Multi-Day Trip Cluster

This repository contains a FastAPI-based project that performs clustering using a custom K-means implementation in TensorFlow and HDBScan for clustering and recommendation. It also includes location scheduling based on proximity between place.

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
- **Clustering and Recommendation**: HDBScan.
- **Location Scheduling**: Allocates time slots for visiting clustered locations within business hours.

## Prerequisites

- **Python 3.12**
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
   conda create --name cluster_env python=3.12
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
├── model/
│   ├── jatim.pkl          
│   ├── bali.pkl         
│   ├── jateng.pkl           
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
        {"name": "Surabaya Zoo", "coordinates": [-7.3024, 112.7367]},
        {"name": "House of Sampoerna", "coordinates": [-7.2482, 112.7356]},
        {"name": "Submarine Monument", "coordinates": [-7.2656, 112.7461]},
        {"name": "Tugu Pahlawan", "coordinates": [-7.2458, 112.7374]},
        {"name": "Ciputra Waterpark", "coordinates": [-7.3167, 112.6308]},
        {"name": "Kenjeran Beach", "coordinates": [-7.2488, 112.8058]},
        {"name": "Galaxy Mall Surabaya", "coordinates": [-7.2940, 112.7700]},
        {"name": "Suroboyo Bridge", "coordinates": [-7.2475, 112.7802]},
        {"name": "Suro and Boyo Statue", "coordinates": [-7.3053, 112.7385]},
        {"name": "Pakuwon Mall", "coordinates": [-7.2916, 112.6429]}
    ],
    "num_clusters": 3,
    "province": "jawa timur",
    "daily_start_time": "08:00",
    "daily_end_time": "18:00"
}
```

### Response

```json
{
  "grouped_clusters": [
    {
      "cluster": 0,
      "avg_duration": 109,
      "schedule": [
        {
          "name": "House of Sampoerna",
          "avg_duration": 109,
          "travel_time": null,
          "mode": null
        },
        {
          "name": "Tugu Pahlawan",
          "avg_duration": 109,
          "travel_time": 6,
          "mode": "walking"
        },
        {
          "name": "Submarine Monument",
          "avg_duration": 109,
          "travel_time": 9,
          "mode": "driving"
        },
        {
          "name": "Suroboyo Bridge",
          "avg_duration": 109,
          "travel_time": 19,
          "mode": "driving"
        },
        {
          "name": "Kenjeran Beach",
          "avg_duration": 109,
          "travel_time": 9,
          "mode": "driving"
        }
      ]
    },
    {
      "cluster": 1,
      "avg_duration": 189,
      "schedule": [
        {
          "name": "Surabaya Zoo",
          "avg_duration": 189,
          "travel_time": null,
          "mode": null
        },
        {
          "name": "Suro and Boyo Statue",
          "avg_duration": 189,
          "travel_time": 15,
          "mode": "walking"
        },
        {
          "name": "Galaxy Mall Surabaya",
          "avg_duration": 189,
          "travel_time": 16,
          "mode": "driving"
        }
      ]
    },
    {
      "cluster": 2,
      "avg_duration": 292,
      "schedule": [
        {
          "name": "Ciputra Waterpark",
          "avg_duration": 292,
          "travel_time": null,
          "mode": null
        },
        {
          "name": "Pakuwon Mall",
          "avg_duration": 292,
          "travel_time": 15,
          "mode": "driving"
        }
      ]
    }
  ],
  "final_unvisitable": []
}
```

Use this example JSON to test the `/recommend/` endpoint:

```json
{
    "points": [
        {"name": "Surabaya Zoo", "coordinates": [-7.3024, 112.7367]},
        {"name": "House of Sampoerna", "coordinates": [-7.2482, 112.7356]},
        {"name": "Submarine Monument", "coordinates": [-7.2656, 112.7461]},
        {"name": "Tugu Pahlawan", "coordinates": [-7.2458, 112.7374]},
        {"name": "Ciputra Waterpark", "coordinates": [-7.3167, 112.6308]},
        {"name": "Kenjeran Beach", "coordinates": [-7.2488, 112.8058]},
        {"name": "Galaxy Mall Surabaya", "coordinates": [-7.2940, 112.7700]},
        {"name": "Suroboyo Bridge", "coordinates": [-7.2475, 112.7802]},
        {"name": "Suro and Boyo Statue", "coordinates": [-7.3053, 112.7385]},
        {"name": "Pakuwon Mall", "coordinates": [-7.2916, 112.6429]}
    ],
    "num_clusters": 3,
    "province": "jawa timur",
    "daily_start_time": "08:00",
    "daily_end_time": "18:00"
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
          "name": "Ciputra Waterpark",
          "avg_duration": 292,
          "travel_time": null,
          "mode": null
        },
        {
          "name": "Pakuwon Mall",
          "avg_duration": 292,
          "travel_time": 15,
          "mode": "driving"
        }
      ],
      "avg_duration": 292
    },
    {
      "cluster": 1,
      "schedule": [
        {
          "name": "Surabaya Zoo",
          "avg_duration": 58,
          "travel_time": null,
          "mode": null
        },
        {
          "name": "Suro and Boyo Statue",
          "avg_duration": 58,
          "travel_time": 15,
          "mode": "walking"
        },
        {
          "name": "Galaxy Mall Surabaya",
          "avg_duration": 58,
          "travel_time": 16,
          "mode": "driving"
        },
        {
          "name": "Submarine Monument",
          "avg_duration": 58,
          "travel_time": 18,
          "mode": "driving"
        },
        {
          "name": "House of Sampoerna",
          "avg_duration": 58,
          "travel_time": 10,
          "mode": "driving"
        },
        {
          "name": "Tugu Pahlawan",
          "avg_duration": 58,
          "travel_time": 6,
          "mode": "walking"
        },
        {
          "name": "Suroboyo Bridge",
          "avg_duration": 58,
          "travel_time": 16,
          "mode": "driving"
        },
        {
          "name": "Kenjeran Beach",
          "avg_duration": 58,
          "travel_time": 9,
          "mode": "driving"
        }
      ],
      "avg_duration": 58
    }
  ],
  "final_unvisitable": [],
  "recommended_days": 2
}
```

# Bonus (Optional)

**Uncomment** the visualization and response in routes.py to gain a better understanding of the clustering and the routing (scheduling) response (or you could check out the static folder)

<img src="https://github.com/Gumm11/Multi-Day-Cluster/blob/main/static/cluster_plot.png" width="600" />
<img src="https://github.com/Gumm11/Multi-Day-Cluster/blob/main/static/routing_plot.png" width="600" />
<img src="https://github.com/Gumm11/Multi-Day-Cluster/blob/main/static/schedule_table_cluster_0.png" width="600" />
<img src="https://github.com/Gumm11/Multi-Day-Cluster/blob/main/static/schedule_table_cluster_1.png" width="600" />
<img src="https://github.com/Gumm11/Multi-Day-Cluster/blob/main/static/schedule_table_cluster_2.png" width="600" />
