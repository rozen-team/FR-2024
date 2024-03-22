from dataclasses import dataclass
import requests
from typing import List, Tuple, Union

@dataclass
class Building:
    coords: Tuple[float]
    target_height: int = None
    real_height: int = None

    def to_dict(self): 
        return {
            "coords": self.coords,
            "target_height": self.target_height,
            "real_height": self.real_height,
            "match": self.real_height == self.target_height
        }

@dataclass
class Border:
    start: Tuple[int]
    end: Tuple[int]

    def to_dict(self):
        return {
            "start": self.start,
            "end": self.end
        }

@dataclass
class Segment:
    border: Border
    length: int

    def to_dict(self):
        return {
            "border": self.border.to_dict(),
            "length": self.length
        }

@dataclass
class Worker:
    coords: Tuple[float]
    is_working: bool

    def to_dict(self):
        return {
            "coords": self.coords,
            "is_working": self.is_working
        }

@dataclass
class Road:
    length: int
    segments: List[Segment]
    workers: List[Worker]

    def to_dict(self):
        return {
            "length": self.length,
            "segments": [s.to_dict() for s in self.segments],
            "workers": [w.to_dict() for w in self.workers]
        }


class RESTClient:
    def __init__(self, token, server_uri = "http://65.108.156.108"):
        self.token = token
        self.server_uri = server_uri

    def post_buildings(self, buildings: List[Building]) -> Tuple[int, Union[List[Building], None]]:
        rest_url = self.server_uri + "/buildings"

        resp = requests.post(
            rest_url,
            json=[b.to_dict() for b in buildings],
            headers={
                "Authorization": f"Bearer {self.token}"
            }
        )

        if resp.status_code == 200:
            json = resp.json()
            b = Building(**json)
            return 200, b

        return resp.status_code, None

    def post_roads(self, roads: List[Road]) -> int:
        rest_url = self.server_uri + "/roads"
        resp = requests.post(
            rest_url,
            json=[r.to_dict() for r in roads],
            headers={
                "Authorization": f"Bearer {self.token}"
            }
        )

        return resp.status_code