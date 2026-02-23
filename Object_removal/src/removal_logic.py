


import os
import cv2
from config import N_FRAMES_MISSING

class RemovalLogic:

    def __init__(self, registry):
        self.registry = registry

    def process(self, visible_ids, frame):

        alerts = []

        for obj_id, data in list(self.registry.data.items()):

            if obj_id not in visible_ids:
                data["missing"] += 1
            else:
                data["missing"] = 0

            if data["missing"] > N_FRAMES_MISSING:

                # CASE 3
                if data["status"] == "abandoned":
                    label = "Resolved"
                    folder = "alerts/resolved"

                # CASE 1
                elif data["person_nearby"]:
                    label = "Removed By Person"
                    folder = "alerts/normal"

                # CASE 2
                else:
                    label = "Suspicious Removal"
                    folder = "alerts/suspicious"

                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(f"{folder}/{obj_id}.jpg", frame)

                alerts.append((obj_id, label))
                self.registry.remove(obj_id)

        return alerts