


# src/abandoned_logic.py

import time
import math
from config import (
    STATIC_TIME_THRESHOLD,
    PIXEL_MOVEMENT_THRESHOLD,
    PERSON_RADIUS_FACTOR
)

class AbandonedLogic:

    def __init__(self):
        self.object_memory = {}

    def _distance(self, p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def update(self, tracked_objects, detections):

        persons = []
        objects = []

        for det in detections:
            if det["class"] == "person":
                persons.append(det)
            else:
                objects.append(det)

        alerts = []

        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centroid = (cx, cy)

            obj_id = f"{cx}_{cy}"

            if obj_id not in self.object_memory:
                self.object_memory[obj_id] = {
                    "centroid": centroid,
                    "static_start": time.time(),
                    "abandoned": False
                }

            prev_centroid = self.object_memory[obj_id]["centroid"]
            movement = self._distance(prev_centroid, centroid)

            if movement > PIXEL_MOVEMENT_THRESHOLD:
                self.object_memory[obj_id]["static_start"] = time.time()

            self.object_memory[obj_id]["centroid"] = centroid

            static_time = time.time() - self.object_memory[obj_id]["static_start"]

            unattended = True

            for person in persons:
                px1, py1, px2, py2 = person["bbox"]
                pcx = (px1 + px2) // 2
                pcy = (py1 + py2) // 2

                radius = (x2 - x1) * PERSON_RADIUS_FACTOR

                if self._distance((pcx, pcy), centroid) < radius:
                    unattended = False
                    break

            if static_time >= STATIC_TIME_THRESHOLD and unattended:
                self.object_memory[obj_id]["abandoned"] = True
                alerts.append((obj, static_time))

        return alerts