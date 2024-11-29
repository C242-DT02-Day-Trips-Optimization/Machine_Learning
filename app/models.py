from pydantic import BaseModel
from typing import List, Optional

class Location(BaseModel):
    name: str
    coordinates: List[float]
    
    def __hash__(self):
        return hash((self.name, tuple(self.coordinates)))
    
    def __eq__(self, other):
        return isinstance(other, Location) and self.name == other.name and self.coordinates == other.coordinates

class ClusteringInput(BaseModel):
    points: List[Location]
    num_clusters: int
    province: Optional[str]
    # Default to 08:00
    daily_start_time: Optional[str] = "08:00"  
    # Default to 20:00
    daily_end_time: Optional[str] = "20:00"    